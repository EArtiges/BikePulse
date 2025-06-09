import pandas as pd
import geopandas as gpd

from shapely import Point
from shapely.ops import transform
from matplotlib import pyplot as plt

# input_raster = "data/global/GHS_POP/R3_C19.tif"
#from raster2xyz.raster2xyz import Raster2xyz
# or use like this: raster2xyz input.tif output.csv
# rtxyz = Raster2xyz()
# rtxyz.translate(input_raster, out_csv)

out_csv = "data/global/GHS_POP/R3_C19.csv"
out_file = "data/global/GHS_POP/R3_C19.gpkg"


df = pd.read_csv(out_csv)
df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']), axis=1)
gdf = gpd.GeoDataFrame(df)
gdf = gdf.set_crs("ESRI:54009")
gdf = gdf.to_crs("epsg:4326")
gdf = gdf.rename(columns={'z':'population'})
gdf.to_file(out_file)
