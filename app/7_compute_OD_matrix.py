import pandas as pd
import geopandas as gpd
from shapely.ops import Point
from scipy.spatial import distance_matrix
import oslo_lib

city='Oslo'
city_crs = "epsg:32632"
bike_trips_path = f'data/{city}/trips/trips.pkl'
hex_grid_path = f'data/{city}/{city.lower()}_grid.geoparquet'
stations_path = f'data/{city}/trips/stations.pkl'

hex_grid = gpd.read_parquet(hex_grid_path)
stations = pd.read_pickle(stations_path)
stations.index.set_names("station_id", inplace=True)

stations['geometry'] = stations.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
stations = gpd.GeoDataFrame(stations, crs='epsg:4326')

def find_cell(station, hex_grid=hex_grid):
    return hex_grid[hex_grid.contains(station)].index[0]

stations['cell_id'] = stations['geometry'].apply(find_cell)

stations_GPS = stations.to_crs(city_crs)['geometry'].apply(lambda x : [x.x, x.y]).to_list()
station_distances = pd.DataFrame(index=stations.index, 
                                 columns=stations.index, 
                                 data=distance_matrix(stations_GPS, stations_GPS)
                                ).astype(int)
station_distances.to_pickle(f'data/{city}/trips/station_distances.pkl')

cell_GPS = hex_grid.to_crs(city_crs).centroid.apply(lambda x : [x.x, x.y]).to_list()
cell_distances = pd.DataFrame(index=hex_grid.index, 
                                 columns=hex_grid.index, 
                                 data=distance_matrix(cell_GPS, cell_GPS)
                                ).astype(int)
cell_distances.to_pickle(f'data/{city}/trips/cell_distances.pkl')

trips = oslo_lib.get_processed_trips(pd.read_pickle(f'data/{city}/trips/trips.pkl'), station_distances)
trips = trips.join(stations['cell_id'].rename('start_cell'), on='start_station_id')
trips = trips.join(stations['cell_id'].rename('end_cell'), on='end_station_id')

trips.to_pickle(f'data/{city}/trips/processed_trips.pkl')

trips_per_station = oslo_lib.get_trips_per_station(trips, stations)

departures_per_station = trips.groupby(['start_station_id', 'year', 'month', 'weekday', 'hour']).trip.sum().rename("departures_per_station")
arrivals_per_station = trips.groupby(['end_station_id', 'year', 'month', 'weekday', 'hour']).trip.sum().rename("arrivals_per_station")
potentials = pd.concat([departures_per_station, arrivals_per_station], axis=1).fillna(0)
potentials.index.set_names('station_id', level=0, inplace=True)
potentials = potentials.join(stations['cell_id'])
potentials.to_pickle(f"data/{city}/trips/potentials.pkl")

OD_matrix = trips.groupby(['year', 'month', 'weekday', 'quarter', 'hour', 'start_cell', 'end_cell'])['trip'].sum()
OD_matrix.to_pickle(f'data/{city}/cell_OD.pkl')
