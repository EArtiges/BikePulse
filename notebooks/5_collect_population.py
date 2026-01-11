import geopandas as gpd

from utils import get_config

city = "Edinburgh"
config = get_config(city)

full_city_name = config["collect_area"]["name"]
city_crs = f"epsg:{config['POIs']['crs']}"

# Get hex grid
hex_grid = gpd.read_parquet(f"data/{city}/hex_grid.geoparquet")
grid_envelope = hex_grid.union_all().envelope
buffer_size = grid_envelope.area**0.5
clip_mask = grid_envelope.buffer(buffer_size)

# Get pop data
pop = gpd.read_file("data/global/GHS_POP/R3_C19.gpkg", mask=clip_mask).to_crs("ESRI:54009")
pop["geometry"] = pop.buffer(50).envelope

# Project both to city CRS
pop = pop.to_crs(city_crs)
hex_grid = hex_grid.to_crs(city_crs)

# Clip areas
pop = pop.clip(hex_grid.union_all())

# get population density
value = "population"
pop[f"{value}_density"] = pop[value] / pop["geometry"].area


def get_cell_pop(cell, pop=pop, value=value):
    """
    Multiply overlap area by pop density and sum across hex cells
    to get hex cell population
    """
    clipped_data = pop.clip(cell)
    return (clipped_data.area * clipped_data[f"{value}_density"]).sum()


hex_grid["population"] = hex_grid["geometry"].apply(get_cell_pop)
hex_grid["population"].to_csv(f"data/{city}/population.csv")
