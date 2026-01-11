"""Tensor decomposition algorithms."""
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac, tucker


def prepare_od_tensor(
    od_matrix: pd.DataFrame,
    weekdays_only: bool = True,
    month_range: Tuple[int, int] = (1, 12),
    normalize_by_year: bool = True,
) -> Tuple[tl.tensor, list, list]:
    """
    Prepare OD matrix as tensor for factorization.

    Parameters
    ----------
    od_matrix : pd.DataFrame
        Origin-destination matrix with MultiIndex (year, month, weekday, hour, start_cell, end_cell)
    weekdays_only : bool
        If True, filter for weekdays (0-4); if False, filter for weekends (5-6)
    month_range : Tuple[int, int]
        (min_month, max_month) to include in analysis
    normalize_by_year : bool
        If True, normalize by total trips per year before aggregating

    Returns
    -------
    Tuple[tl.tensor, list, list]
        - Tensor (hours × start_cells × end_cells)
        - List of hours
        - List of cells (ordered)
    """
    # Drop future years if present
    od_matrix = od_matrix.drop(2025, level=0, errors="ignore")

    # Normalize by year if requested
    if normalize_by_year:
        total_trips_per_year = od_matrix.groupby(level=0).sum()
        od_matrix = od_matrix / total_trips_per_year
        # Average across all years
        od_matrix = od_matrix.groupby(level=(1, 2, 3, 4, 5)).mean()
    else:
        od_matrix = od_matrix.groupby(level=(1, 2, 3, 4, 5)).sum()

    od_matrix = od_matrix.reset_index()

    # Filter by weekday
    if weekdays_only:
        od_matrix = od_matrix.query("weekday < 5")
    else:
        od_matrix = od_matrix.query("weekday >= 5")

    # Drop weekday column
    od_matrix = od_matrix.drop("weekday", axis=1, errors="ignore")

    # Filter by month
    min_month, max_month = month_range
    od_matrix = od_matrix.query(f"month >= {min_month} & month <= {max_month}")

    # Aggregate by hour, start_cell, end_cell
    dataset = (
        od_matrix.groupby(["hour", "start_cell", "end_cell"])
        .trip.mean()
        .reset_index()
    )

    # Pivot to matrix form
    dataset = dataset.pivot(
        index=["hour", "start_cell"], columns="end_cell", values="trip"
    ).fillna(0)

    # Normalize to percentage
    dataset = 100 * dataset / dataset.sum().sum()

    # Convert to tensor
    hours = sorted(dataset.index.get_level_values(0).unique())
    time_slices = [dataset.loc[h] for h in hours]
    cells = time_slices[0].columns.tolist()

    tensor_array = np.array(
        [ts.reindex(ts.columns).fillna(0).values for ts in time_slices]
    )

    tensor = tl.tensor(data=tensor_array.astype("float64"))

    return tensor, hours, cells


def factorize_tensor(
    tensor: tl.tensor,
    method: Literal["tucker", "parafac"] = "tucker",
    rank: int | Tuple[int, int, int] = (3, 5, 5),
    random_seed: int = 42,
    init: str = "svd",
) -> dict:
    """
    Perform tensor factorization.

    Parameters
    ----------
    tensor : tl.tensor
        Input tensor (hours × start_cells × end_cells)
    method : str
        Factorization method: 'tucker' or 'parafac'
    rank : int or Tuple[int, int, int]
        For Tucker: (temporal_rank, spatial_source_rank, spatial_dest_rank)
        For PARAFAC: single integer rank
    random_seed : int
        Random seed for reproducibility
    init : str
        Initialization method ('svd' or 'random')

    Returns
    -------
    dict
        Dictionary with factorization results
    """
    tl.set_backend("numpy")
    np.random.seed(random_seed)

    if method == "tucker":
        if isinstance(rank, int):
            rank = (rank, rank, rank)
        core, factors = tucker(tensor, rank=rank, init=init, random_state=random_seed)
        return {
            "method": "tucker",
            "core": core,
            "factors": factors,
            "rank": rank,
        }
    elif method == "parafac":
        if isinstance(rank, tuple):
            rank = rank[0]
        result = parafac(tensor, rank=rank, init=init, random_state=random_seed)
        return {
            "method": "parafac",
            "factors": result.factors,
            "weights": result.weights,
            "rank": rank,
        }
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'tucker' or 'parafac'")


def reconstruct_tensor(factorization_result: dict) -> tl.tensor:
    """
    Reconstruct tensor from factorization.

    Parameters
    ----------
    factorization_result : dict
        Result from factorize_tensor

    Returns
    -------
    tl.tensor
        Reconstructed tensor
    """
    method = factorization_result["method"]

    if method == "tucker":
        from tensorly.tucker_tensor import tucker_to_tensor

        return tucker_to_tensor(
            (factorization_result["core"], factorization_result["factors"])
        )
    elif method == "parafac":
        from tensorly.cp_tensor import cp_to_tensor

        return cp_to_tensor(
            (factorization_result["weights"], factorization_result["factors"])
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_spatial_factors(
    factorization_result: dict, cells: list
) -> pd.DataFrame:
    """
    Extract spatial factors as DataFrame.

    Parameters
    ----------
    factorization_result : dict
        Result from factorize_tensor
    cells : list
        Ordered list of cell IDs

    Returns
    -------
    pd.DataFrame
        Spatial factors with columns: start_cell, end_cell, component, trips
    """
    factors_list = factorization_result["factors"]

    if factorization_result["method"] == "tucker":
        temporal_factors, W, H = factors_list
    else:  # parafac
        temporal_factors, W, H = factors_list

    n_components = W.shape[1]

    all_factors = []
    for n in range(n_components):
        X_ = np.outer(W[:, n], H[:, n])
        factor_df = pd.DataFrame(
            X_, index=pd.Index(cells, name="start_cell"), columns=cells
        )
        factor_df.columns.name = "end_cell"
        factor_df = factor_df.stack().rename("trips").reset_index()
        factor_df["component"] = n
        all_factors.append(factor_df)

    factors = pd.concat(all_factors, ignore_index=True)

    # Add full factor (sum of all components)
    full_factor = (
        factors.groupby(["start_cell", "end_cell"])
        .trips.sum()
        .rename("trips")
        .reset_index()
    )
    full_factor["component"] = "full"
    factors = pd.concat([factors, full_factor], ignore_index=True)

    return factors
