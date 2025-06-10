import oslo_lib
import pandas as pd

city = 'Edinburgh'
providers = {'Oslo': 'oslobysykkel.no', 'Edinburgh': 'edinburghcyclehire.com', 'Milan':'bikemi.com'}
provider = providers[city]

df = oslo_lib.collect_data(years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], months = range(1, 13), provider=provider)
df.to_pickle(f'data/{city}/trips/{city.lower()}_data.pkl')

start_stations = df.set_index('start_station_id')[['start_station_longitude', 'start_station_latitude', 'start_station_name', 'start_station_description']]
start_stations.columns = ['longitude', 'latitude', 'name', 'description']

end_stations = df.set_index('end_station_id')[['end_station_longitude', 'end_station_latitude', 'end_station_name', 'end_station_description']]
end_stations.columns = ['longitude', 'latitude', 'name', 'description']

stations = pd.concat([start_stations,end_stations])
stations = stations[~stations.index.duplicated()].sort_index()

stations.to_pickle(f'data/{city}/trips/stations.pkl')

trips = df[['started_at', 'ended_at', 'start_station_id', 'end_station_id', 'duration']].copy()
trips.to_pickle(f'data/{city}/trips/trips.pkl')