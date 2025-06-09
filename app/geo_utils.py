import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

ox.settings.use_cache=True
ox.settings.log_console=True

def get_boundaries(array):
    return min(array), max(array)

def get_spread(boundaries):
    return max(boundaries) - min(boundaries)

def get_center(spread):
    return np.mean(spread)

def get_window(center, spread, buffer):
    buffer = buffer*spread/2
    return (center - spread/2 - buffer), (center + spread/2 + buffer)

def get_windows(latitudes, longitudes, buffer):

    longitude_spread = get_spread(longitudes)
    longitude_center = get_center(longitudes)
    longitude_window = get_window(longitude_center, longitude_spread, buffer)

    latitude_spread = get_spread(latitudes)
    latitude_center = get_center(latitudes)
    latitude_window = get_window(latitude_center, latitude_spread, buffer)
    
    return latitude_window, longitude_window

def window_filter(series, window):
    f1 = series >= window[0]
    f2 = series <= window[1]    
    return (f1) & (f2)

def get_city_bbox(city_area: gpd.GeoDataFrame):
    return city_area.bounds.values[0]
    
def geo_filter(df, latitude_window, longitude_window):
    lat_filter = window_filter(df.latitude, latitude_window)
    lon_filter = window_filter(df.longitude, longitude_window)
    return (lat_filter) & (lon_filter)