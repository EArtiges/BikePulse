"""
Takes in an area file, reads it, build the relevant hex grid and stores the result.
"""

from functools import partial

import geopandas as gpd

from utils import get_config

city = "Edinburgh"
config = get_config(city)
city_crs = config["POIs"]["crs"]
stations = gpd.read_parquet(config["collect_area"]["filename"])
hull = gpd.GeoSeries(stations.union_all().convex_hull, name="geometry", crs="epsg:4326")
buffed_hull = hull.to_crs(city_crs).buffer(1000).to_crs("epsg:4326").rename("geometry").to_frame()
hex_grid = buffed_hull.h3.polyfill_resample(config["grid"]["resolution"])


def find_neighbours(cell_index, n_neighbours=1, hex_grid=hex_grid):
    cell = hex_grid.loc[cell_index, "geometry"]
    neighbours = hex_grid[hex_grid.intersects(cell)].index
    if n_neighbours == 1:
        yield from neighbours
    else:
        for neighbour in neighbours:
            yield from find_neighbours(neighbour, n_neighbours - 1)


def get_area(cell_index, n_neighbours, hex_grid=hex_grid):
    return hex_grid.loc[find_neighbours(cell_index, n_neighbours, hex_grid)].union_all()


neighbours_functions = {
    "first_neighbours": partial(get_area, n_neighbours=1),
    "second_neighbours": partial(get_area, n_neighbours=2),
    "third_neighbours": partial(get_area, n_neighbours=3),
}

for key, func in neighbours_functions.items():
    hex_grid[key] = hex_grid.index.to_series().map(func).astype(hex_grid["geometry"].dtype)

hex_grid.to_parquet(config["grid"]["full_grid_filename"])

stations = gpd.read_parquet(f"data/{city}/trips/stations.geoparquet").to_crs(hex_grid.crs)
stations_locations = stations.union_all().convex_hull

clipped_grid = hex_grid[hex_grid.intersects(stations_locations)]
clipped_grid.to_parquet(config["grid"]["clipped_grid_filename"])
