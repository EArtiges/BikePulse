import geopandas as gpd
import pandas as pd
from shapely import Point

from utils import get_config

# Before running this script ocnvert the tif into a csv from the terminal:
#! raster2xyz input.tif output.csv

city = "Edinburgh"
config = get_config(city)

df = pd.read_csv(config["population"]["input"])
df["geometry"] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
gdf = gpd.GeoDataFrame(df)
gdf = gdf.set_crs("ESRI:54009")
gdf = gdf.to_crs("epsg:4326")
gdf = gdf.rename(columns={"z": "population"})
gdf.to_file(config["population"]["output"])
