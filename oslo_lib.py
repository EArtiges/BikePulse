import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import Point
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_tucker


def obtain_url(y, m):
    assert m>=1 and m<=12
    if m < 10:
        m=f"0{m}"
    return f"https://data.urbansharing.com/oslobysykkel.no/trips/v1/{y}/{m}.json"

def urls(years, months):
    for y in years:
        for m in months:
            if (y==2019 and m<4) or (y==2023 and m>7):
                pass
            else:
                yield obtain_url(y, m)

def retrieve_dataset(url):
    r = requests.get(url)
    json_content = r.json()
    df = pd.DataFrame(json_content)
    
    if len(df)>0:    
        print(f'no data for {url}')
        df.started_at = pd.to_datetime(df.started_at)
        df.ended_at = pd.to_datetime(df.ended_at)
        df.start_station_id = df.start_station_id.astype(int)
        df.end_station_id = df.end_station_id.astype(int)
        
    return df

def collect_data(years, months):
    
    global_df = pd.DataFrame()
    
    for url in urls(years, months):
        print(url)
        df = retrieve_dataset(url)
        global_df = pd.concat([global_df, df])
    
    return global_df

def get_processed_trips(trips, station_distances):

    trips['date'] = trips.started_at.dt.date

    trips['year'] = trips.started_at.apply(lambda x : x.year)
    trips['month'] = trips.started_at.apply(lambda x : x.month)
    trips['weekday'] = trips.started_at.apply(lambda x : x.weekday())
    trips['weekday_name'] = trips.started_at.apply(lambda x : x.day_name())
    trips['weekend'] = trips.weekday>4

    trips['hour'] = trips.started_at.apply(lambda x : x.hour)

    trips['quarter'] = trips.started_at.apply(lambda x : x.hour // 3)
    quarter_map = {i//3:f"{i} - {i+3}" for i in range(0, 24, 3)}
    trips['quarter_name'] = trips.quarter.map(quarter_map)

    trips['days_in_month'] = trips.started_at.apply(lambda x : x.days_in_month)
    trips['week'] = trips.started_at.apply(lambda x : x.week)

    trips['trip'] = 1

    trips = trips.join(station_distances.stack().rename('distance'), on = ['start_station_id', 'end_station_id'])

    trips['duration_min'] = trips.duration // 60
    trips['distance_km'] = trips.distance // 1000

    trips['speed'] = trips.distance/trips.duration  # m/s
    trips['speed_kmh'] = 3.6 * trips.speed  # km/h
    
    return trips


def get_geostations(stations):
    stations['geometry'] = stations.apply(lambda x : Point(x.longitude, x.latitude), axis=1)
    stations = gpd.GeoDataFrame(stations)
    stations = stations.set_crs('epsg:4326')
    stations = stations.to_crs('epsg:27393')
    return stations


def get_trips_per_station(trips, stations):
    
    starts = trips.groupby('start_station_id').trip.sum().reindex(stations.index).fillna(0)
    ends = trips.groupby('end_station_id').trip.sum().reindex(stations.index).fillna(0)
    trips_per_station =  starts + ends
    
    trips_per_station = stations.join(trips_per_station)
    
    def station_class(x):
        if x<6e4:
            return 0
        elif x<8e4:
            return 1
        elif x<12e4:
            return 2
        else:
            return 3

    trips_per_station['station_class'] = trips_per_station.trip.apply(station_class)
    
    return trips_per_station

def get_stations_GPS(trips_per_station):
        return trips_per_station.geometry.apply(lambda x : [x.x, x.y]).to_list()
    
def filter_f(f, top_trips, threshold_function, q_thresh):
    if top_trips:
        f = f.sort_values(by='trips', ascending=False).head(top_trips)
    else:
        if threshold_function:
            threshold = threshold_function(f.trips.values)
        elif q_thresh:
            threshold  = f.trips.quantile(q_thresh)
        else:
            threshold = 0
        f = f[f.trips > threshold]
    return f

def get_matrix(data, clusters, sample_size=None):
    
    full_index = pd.MultiIndex.from_product([range(24), range(clusters.cluster.max()+1)])
    
    if sample_size:
        data_sample = data.sample(sample_size)
    else:
        data_sample = data
    matrix = data.groupby(['hour', 'start_cluster', 'end_cluster']).trip.sum().unstack().fillna(0).astype(int).reindex(full_index)
    rev_matrix = data.groupby(['hour', 'end_cluster', 'start_cluster']).trip.sum().unstack().fillna(0).astype(int).reindex(full_index)

    matrix.index.names = ['hour', 'cluster']
    rev_matrix.index.names = ['hour', 'cluster']
    matrix.columns.name = 'cluster'
    rev_matrix.columns.name = 'cluster'

    matrix = (matrix + rev_matrix).fillna(0)
    for i in matrix.loc[0].index:
        if i not in matrix.columns:
            matrix[i] = 0
    matrix.sort_index(axis=1, inplace=True)
    
    return matrix


def get_T(matrix):
    return tl.tensor(data = [tl.tensor(matrix.loc[h]).astype('float64') for h in range(24)])

def bootstrap_T(matrix, 
                frac, 
                data,
                clusters
               ):
    matrix_sample = get_matrix(data, clusters, int(frac*len(data)))
    return get_T(matrix_sample)

def get_factorization(use):
    if use == 'tucker':
        factorization = non_negative_tucker
        reconstruction_function = tl.tucker_to_tensor
    elif use == 'parafac':
        factorization = non_negative_parafac
        reconstruction_function = tl.cp_to_tensor
    return factorization, reconstruction_function
