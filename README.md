1. Collect bike trips
  - Given a storage location, list all files
  - Download files and store them somewhere (in `data/iron`)
  - Otherwise plug into RSS feed for bikes leaving/arriving at stations
  - Listen and collect for a while
2. Preprocess trips
  - For each location write a preprocessing function
  - Store preprocessed data somewhere (in `data/silver`)
3. Collect area
  - For each area get a name and a CRS projection
  - For each area get the corresponding Overture GERS id -> `config`
  - download the corresponding shapefile -> `data/shapefiles`
  - Retrieve the H3 grid at the desired resolution and crop it
4. Collect place data
  - For each area collect: 
    - POIs using Overture
    - Building height and density using Microsoft
    - Canopy cover using MODIS
    - Street network using OSM or more recet packages
    - store the whole thing in `data/iron`
5. Engineer features
  - For each area, engineer features for cell classification and trip prediction
    - count of POIs/volume of building/length of bike path in the cell, around the cell, distance weighted... etc.
    - get the departure and arrival profiles per cell throughout the day
  - Store results in `data/silver`
6. Grid-search train models:
  1. cell typology clustering (unsure)
    - We need a global cell typology, probably this will be established using clustering on collected cells and by manually labelling clusters. This typology can be updated regularly and being tracked.
    - Can this typology really work from one city to the next? Business district in NYC and in Abidjan will look very different.
  2. Cell type classifier
  3. Average day
    - Using real trips, train a model to estimate average day in terms of bikes out/bikes in per cell using partial data (i.e only a week of winter data or only a month of spring)
  4. Trip predictor: predict trips between a departure and an arrival cell based on cell features and typical day profile
    - global model
    - specific model per location
  5. Trip factorization
    - Run on real data if we have it
    - Run on fake data using trip predictor otherwise
    - Follow chatGPT idea to cross-validate factorization and end up woth something robust
  6. Factor description
    - Using cells descriptions and factor data, use LLM to generate a description of each factor
7. Display!
  1. Map the area
  2. Display cells
  3. Map departures and arrivals per cell
  4. Display high-level stats and cell temporal profiles
  5. Display factors + explanations
  6. Let the user place more POIs in cells: a new museum, a new bar, etc. and recompute trips using pre-trained models. Display change in number of trips per cell.
  

