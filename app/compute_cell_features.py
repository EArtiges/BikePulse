import geopandas as gpd
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

config_path = "app/run.yml"
with open(config_path) as file:
    config = safe_load(file)

folder = config['POIs']['folder']
city_crs = f"epsg:{config['POIs']['crs']}"

cycleways = gpd.read_parquet(folder+'cycleways.geoparquet')
land_use = gpd.read_parquet(folder+'land_use.geoparquet')
water = gpd.read_parquet(folder+'water.geoparquet')
bus_stations = gpd.read_parquet(folder+'bus_stations.geoparquet')
railway_stations = gpd.read_parquet(folder+'railway_stations.geoparquet')
subway_stations = gpd.read_parquet(folder+'subway_stations.geoparquet')
other_stations = gpd.read_parquet(folder+'other_stations.geoparquet')
population = pd.read_csv("data/Oslo/oslo_pop.csv").set_index('h3_polyfill')
hex_grid = gpd.read_parquet("data/Oslo/oslo_grid.geoparquet")

cycleways['real_length'] = cycleways.geometry.to_crs(city_crs).length
cycleways = cycleways.explode('osmid').explode('highway')
cycleways = cycleways.drop_duplicates(subset=['osmid', 'highway'])
land_use['area'] = land_use.to_crs(city_crs).area
water['area'] = water.to_crs(city_crs).area

def get_features(cell_index, nth_neighbours='first_neighbours'):
    cell = hex_grid.loc[cell_index, nth_neighbours]
    features = {}
    features |= land_use[land_use.intersects(cell)].groupby('landuse')['area'].sum().to_dict()
    features |= water[water.intersects(cell)].groupby('water')['area'].sum().to_dict()
    features |= cycleways[cycleways.intersects(cell)].groupby('highway')['real_length'].sum().to_dict()

    features['n_cell_bus_stations'] = len(bus_stations[bus_stations.intersects(cell)])
    features['n_cell_railway_stations'] = len(railway_stations[railway_stations.intersects(cell)])
    features['n_cell_subway_stations'] = len(subway_stations[subway_stations.intersects(cell)])
    features['n_cell_other_stations'] = len(other_stations[other_stations.intersects(cell)])
    features['population'] = population.loc[cell_index, 'population']
    return features

features = {
    cell_index: get_features(cell_index)
    for cell_index in hex_grid.index
}
pd.DataFrame(features).T.to_parquet("data/Oslo/cell_features.parquet")

