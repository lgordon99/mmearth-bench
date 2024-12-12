# MMEarth-Bench

## Setup Google Earth Engine
1. Run
```
bash bash-scripts/install_gcloud_CLI.sh
```
2. Go to the [Google Cloud Console](console.cloud.google.com)
3. Create a new project called `mmearth-bench`
4. Go to [Google Earth Engine](https://code.earthengine.google.com/register)
5. Click `Register a Noncommercial or Commercial Cloud project`
6. Select `Unpaid usage`
7. From the dropdown menu choose `Academia & Research`
8. Click `NEXT`
9. Select `Choose an existing Google Cloud Project`
10. From the dropdown menu choose `mmearth-bench`
11. Click `CONTINUE TO SUMMARY`
12. Click `CONFIRM`

## Setup conda environment
1. Open a new terminal
2. Run
```
bash bash-scripts/create_env.sh PATH_TO_ENV_DIR/mmearth-bench-env
```
3. Run
```
conda activate PATH_TO_ENV_DIR/mmearth-bench-env
```
4. Run
```
bash bash-scripts/install_packages.sh
```

## Generate biomass tiles
1. Run
```
python generate_biomass_tiles.py get_biomass_tile_counts
```
which takes around 13 minutes.

2. Run
```
python generate_biomass_tiles.py
```
3. Run
```
python generate_biomass_tiles.py merge_ecoregion_tiles
```

## Generate species tiles
1. Download the [sinr repository](https://github.com/elijahcole/sinr/tree/main)
2. Follow the [instructions for data preparation](https://github.com/elijahcole/sinr/tree/main/data#instructions-for-data-preparation)
3. Rename the `data` folder to `sinr-data`
4. Move `sinr-data` into your data directory for this code located at `data_dir_path`
5. Ensure you have 20GB of memory available
5. Run
```
python generate_species_tiles.py generate_species_tiles
```
which takes around 22 minutes.

6. Run
```
python generate_species_tiles.py make_species_grid
```

## Generate soil tiles
1. Download the [WoSIS December 2023 snapshot](https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip)
2. Move the unzipped folder into your data directory for this code located at `data_dir_path`
3. Run
```
python generate_soil_tiles.py
```

## Download tile data
Run
```
python get_tile_data.py biomass
```
```
python get_tile_data.py species
```
```
python get_tile_data.py soil_nitrogen
```
```
python get_tile_data.py soil_organic_carbon
```
```
python get_tile_data.py soil_pH
```

## Convert data to HDF5
```
python convert_to_h5.py
```