import geopandas as gpd
import pandas as pd
from shapely.ops import Point

city = "Edinburgh"
df = pd.read_pickle(f"data/{city}/trips/{city.lower()}_data.pkl")

start_stations = df.set_index("start_station_id")[
    [
        "start_station_longitude",
        "start_station_latitude",
        "start_station_name",
        "start_station_description",
    ]
]
start_stations.columns = ["longitude", "latitude", "name", "description"]

end_stations = df.set_index("end_station_id")[
    ["end_station_longitude", "end_station_latitude", "end_station_name", "end_station_description"]
]
end_stations.columns = ["longitude", "latitude", "name", "description"]

stations = pd.concat([start_stations, end_stations])
stations = stations[~stations.index.duplicated()].sort_index()
stations.index.set_names("station_id", inplace=True)
stations["geometry"] = stations.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
stations = gpd.GeoDataFrame(stations, crs="epsg:4326")

stations.to_parquet(f"data/{city}/trips/stations.geoparquet")

trips = df[["started_at", "ended_at", "start_station_id", "end_station_id", "duration"]].copy()
trips.to_pickle(f"data/{city}/trips/trips.pkl")
