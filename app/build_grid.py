"""
Takes in an area file, reads it, build the relevant hex grid and stores the result.
"""

import geopandas as gpd
import h3pandas
from yaml import safe_load
from functools import partial

config_path = "app/run.yml"
with open(config_path) as file:
    config = safe_load(file)

hex_grid = gpd.read_parquet(config["collect_area"]['filename'])
hex_grid = hex_grid.h3.polyfill_resample(config['grid']['resolution'])

def find_neighbours(cell_index, n_neighbours=1, hex_grid=hex_grid):
    cell = hex_grid.loc[cell_index, 'geometry']
    neighbours = hex_grid[hex_grid.intersects(cell)].index
    if n_neighbours==1:
        yield from neighbours
    else:
        for neighbour in neighbours:
            yield from find_neighbours(neighbour, n_neighbours-1)

def get_area(cell_index, n_neighbours, hex_grid=hex_grid):
    return hex_grid.loc[find_neighbours(cell_index, n_neighbours, hex_grid)].union_all()


neighbours_functions = {
    "first_neighbours" : partial(get_area, n_neighbours=1),
    "second_neighbours" : partial(get_area, n_neighbours=2),
    "third_neighbours" : partial(get_area, n_neighbours=3)
}

for key, func in neighbours_functions.items():
    hex_grid[key] = hex_grid.index.to_series().map(func).astype(hex_grid['geometry'].dtype)

hex_grid.to_parquet(config['grid']['filename'])
