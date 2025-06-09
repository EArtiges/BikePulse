import geopandas as gpd
from yaml import safe_load
from geo_utils import get_city_bbox
import osmnx as ox
import numpy as np

config_path = "app/run.yml"
with open(config_path) as file:
    config = safe_load(file)

full_city_name = config['collect_area']['name']
city_crs = f"epsg:{config['POIs']['crs']}"

city_boundary = gpd.read_parquet(config["collect_area"]['filename'])
city_bbox = get_city_bbox(city_boundary)

# Get street network 
G_bikes = ox.graph_from_bbox(city_bbox, network_type='bike')
nodes, edges = ox.graph_to_gdfs(G_bikes)
cycleways = edges[edges['highway'].apply(lambda x : 'cycleway' in x)]
cycleways = cycleways[['osmid', 'highway', 'length', 'geometry', 'width']]
for key_col in 'osmid', 'highway', 'width':
    cycleways[key_col] = cycleways[key_col].apply(lambda x: x if isinstance(x, list) else [x])

land_use = ox.features_from_bbox(city_bbox, tags={'landuse':True})
water = ox.features_from_bbox(city_bbox, tags={'natural':'water'})
railway_stations = ox.features_from_bbox(city_bbox, tags={'railway':'station'})
bus_stations = ox.features_from_bbox(city_bbox, tags={'amenity':'bus_station'})

try:
    subway_stations = railway_stations[railway_stations.station=='subway']
    other_stations =  railway_stations[railway_stations.station!='subway']
except:
    subway_stations = railway_stations.head(0)
    other_stations =  railway_stations

folder = config['POIs']['folder']
cycleways.to_parquet(folder+'cycleways.geoparquet')
land_use.to_parquet(folder+'land_use.geoparquet')
water.to_parquet(folder+'water.geoparquet')
bus_stations.to_parquet(folder+'bus_stations.geoparquet')
railway_stations.to_parquet(folder+'railway_stations.geoparquet')
subway_stations.to_parquet(folder+'subway_stations.geoparquet')
other_stations.to_parquet(folder+'other_stations.geoparquet')
