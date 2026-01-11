"""Evaluation metrics for tensor factorization."""
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.metrics import RMSE

from .decomposition import factorize_tensor, reconstruct_tensor


def evaluate_rmse_by_rank(
    tensor: tl.tensor,
    method: str = "parafac",
    rank_range: range = range(1, 31),
    n_runs: int = 10,
    init: str = "svd",
) -> Tuple[List[int], List[float], List[float]]:
    """
    Evaluate RMSE error across different ranks.

    Parameters
    ----------
    tensor : tl.tensor
        Input tensor
    method : str
        Factorization method ('tucker' or 'parafac')
    rank_range : range
        Range of ranks to evaluate
    n_runs : int
        Number of runs per rank (for averaging)
    init : str
        Initialization method

    Returns
    -------
    Tuple[List[int], List[float], List[float]]
        - List of ranks
        - List of mean RMSE errors
        - List of RMSE standard deviations
    """
    errors = []
    error_stds = []

    for rank in rank_range:
        local_errors = []
        for _ in range(n_runs):
            # Each run uses a different random state
            state = np.random.RandomState()
            result = factorize_tensor(
                tensor, method=method, rank=rank, init=init, random_seed=state.randint(10000)
            )
            reconstructed = reconstruct_tensor(result)
            error = RMSE(tensor, reconstructed)
            local_errors.append(error)

        errors.append(np.mean(local_errors))
        error_stds.append(np.std(local_errors))

    return list(rank_range), errors, error_stds


def compute_ccc_for_rank(
    tensor: tl.tensor,
    rank: int,
    method: str = "parafac",
    n_bootstraps: int = 100,
    bootstrap_frac: float = 0.8,
    init: str = "svd",
    use_W: bool = True,
) -> float:
    """
    Compute Consensus Clustering Coefficient for a given rank.

    This is a placeholder - the actual implementation would need
    the CCC module logic which is quite complex.

    Parameters
    ----------
    tensor : tl.tensor
        Input tensor
    rank : int
        Factorization rank
    method : str
        Factorization method
    n_bootstraps : int
        Number of bootstrap samples
    bootstrap_frac : float
        Fraction of data to use in each bootstrap
    init : str
        Initialization method
    use_W : bool
        Whether to use W (source) or H (destination) matrix for consensus

    Returns
    -------
    float
        Consensus clustering coefficient (0-1)
    """
    # This would use the CCC.py module's compute_rho function
    # For now, return a placeholder
    import sys
    from pathlib import Path

    # Import CCC module
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "notebooks"))
    try:
        import CCC

        # Use the existing CCC implementation
        # Note: This requires adapting the interface
        return 0.0  # Placeholder
    except ImportError:
        return 0.0


def compute_optimal_rank(
    rmse_errors: List[float],
    ccc_h_scores: List[float],
    ccc_w_scores: List[float],
) -> Tuple[int, pd.DataFrame]:
    """
    Compute optimal rank using L2 distance metric.

    Parameters
    ----------
    rmse_errors : List[float]
        Normalized RMSE errors (0-1)
    ccc_h_scores : List[float]
        CCC scores for H (destination) matrix
    ccc_w_scores : List[float]
        CCC scores for W (source) matrix

    Returns
    -------
    Tuple[int, pd.DataFrame]
        - Optimal rank
        - DataFrame with all error metrics
    """

    def l2_norm(x):
        """Compute L2 norm."""
        return sum([e**2 for e in x]) ** 0.5

    # Normalize RMSE if not already
    rmse_normalized = [e / max(rmse_errors) for e in rmse_errors]

    # Create DataFrame with all metrics
    df_errors = pd.DataFrame(
        {
            "RMSE": rmse_normalized,
            "CCC_H": ccc_h_scores,
            "CCC_W": ccc_w_scores,
        },
        index=range(1, len(rmse_errors) + 1),
    )

    # Invert CCC (higher is better, but we want error metric)
    df_errors["inv_CCC_H"] = 1 - df_errors["CCC_H"]
    df_errors["inv_CCC_W"] = 1 - df_errors["CCC_W"]

    # Compute L2 errors
    df_errors["error_L2_H"] = df_errors[["RMSE", "inv_CCC_H"]].apply(l2_norm, axis=1)
    df_errors["error_L2_W"] = df_errors[["RMSE", "inv_CCC_W"]].apply(l2_norm, axis=1)
    df_errors["error_L2_global"] = df_errors[["error_L2_H", "error_L2_W"]].apply(
        l2_norm, axis=1
    )

    # Find optimal rank (minimum global error)
    optimal_rank = df_errors["error_L2_global"].idxmin()

    return optimal_rank, df_errors


def compute_reconstruction_metrics(
    tensor: tl.tensor, reconstructed: tl.tensor
) -> Dict[str, float]:
    """
    Compute various reconstruction quality metrics.

    Parameters
    ----------
    tensor : tl.tensor
        Original tensor
    reconstructed : tl.tensor
        Reconstructed tensor

    Returns
    -------
    Dict[str, float]
        Dictionary with metric names and values
    """
    error = tensor - reconstructed

    return {
        "rmse": RMSE(tensor, reconstructed),
        "mae": float(np.mean(np.abs(error))),
        "relative_error_mean": float(np.mean(error / tensor.mean())),
        "relative_error_std": float(np.std(error / tensor.mean())),
        "r_squared": float(
            1 - np.sum(error**2) / np.sum((tensor - tensor.mean()) ** 2)
        ),
    }
