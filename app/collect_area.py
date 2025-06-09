"""
Reads a config file and collects the relevant area from OSMNX. Saves the result under the specified name.
"""
from yaml import safe_load
import osmnx

config_path = "app/run.yml"
with open(config_path) as file:
    config = safe_load(file)

config = config["collect_area"]
city_boundary = osmnx.geocode_to_gdf([config["name"]])
city_boundary.to_parquet(config["filename"])