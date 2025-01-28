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
2. To initialize the Google Cloud CLI and create the conda environment, run
```
bash bash-scripts/create_env.sh PATH_TO_ENV_DIR/mmearth-bench-env
```
3. To activate the conda environment, run
```
conda activate PATH_TO_ENV_DIR/mmearth-bench-env
```
4. To install the needed packages in the conda environment, run
```
bash bash-scripts/install_packages.sh
```

## Generate biomass points
1. To calculate the number of tiles needed in each ecoregion, run
```
python generate_biomass_tiles.py get_biomass_tile_counts
```
which takes around 13 minutes.

2. To generate points in each ecoregion that have GEDI points in their tiles, run
```
python generate_biomass_points.py
```
This generates the folder `biomass/points/ecoregion_points`.

3. To merge all of the ecoregion points into a single file, run
```
python generate_biomass_points.py merge_ecoregion_tiles
```
This generates four files in `biomass/points`.


## Generate species points
1. Download the [sinr repository](https://github.com/elijahcole/sinr/tree/main)
2. Follow the [instructions for data preparation](https://github.com/elijahcole/sinr/tree/main/data#instructions-for-data-preparation)
3. Rename the `data` folder to `sinr-data`
4. Move `sinr-data` into your data directory for this code located at `data_dir_path`
5. Ensure you have 20GB of memory available
6. To generate points that have species observations in their tiles, run
```
python generate_species_points.py
```
which takes around ~1 hour and 10 minutes.

7. Run
```
python generate_species_points.py make_species_grid
```


## Generate soil points
1. Download the [WoSIS December 2023 snapshot](https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip)
2. Move the unzipped folder into your data directory for this code located at `data_dir_path`
3. To generate points that have soil observations in their tiles, run
```
python generate_soil_points.py
```

## Generate tiles
To generate tiles, do the following for each task.

1. To save aligned modality and task data as a tiff for every tile, run
```
python get_tile_data.py TASK
```

2. Ensure you have 50GB memory available. To convert the data to H5 format, run
```
python convert_to_h5.py TASK
```

3. To generate the PNGs to be loaded onto the map, run
```
python generate_map_data.py TASK
```
