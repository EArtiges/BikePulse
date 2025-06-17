"""
Reads a config file and collects the relevant area from OSMNX. Saves the result under the specified name.
"""
import osmnx
from utils import get_config

city = "Edinburgh"
config = get_config(city)
config = config["collect_area"]
city_boundary = osmnx.geocode_to_gdf([config["name"]])
city_boundary.to_parquet(config["filename"])