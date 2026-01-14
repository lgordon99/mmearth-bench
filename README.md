[![Website](https://img.shields.io/badge/🌍-Project_Page-black)](https://lgordon99.github.io/mmearth-bench/)

<h1 align="center">MMEarth-Bench</h1>

<p align="center">
  Lucia Gordon<sup>1,2</sup>, Serge Belongie<sup>2</sup>, Christian Igel<sup>2</sup>, Nico Lang<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup> <img src="docs/static/images/harvard_logo.svg" alt="Harvard" height="70" style="vertical-align: middle;" />
  &nbsp;&nbsp;
  <sup>2</sup> <img src="docs/static/images/ku_logo.png" alt="University of Copenhagen" height="60" style="vertical-align: middle;" />
</p>

<p align="center">
	<img src="docs/static/images/overview_figure.svg" alt="Overview figure" width="500" />
</p>

<details>
<summary><strong>Recreating the dataset</strong></summary>

### Setup Google Earth Engine
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

### Setup conda environment
1. Open a new terminal
2. To initialize the Google Cloud CLI and create the conda environment, run
```
bash bash-scripts/create_env.sh PATH_TO_ENV_DIR
```
3. To activate the conda environment, run
```
conda activate PATH_TO_ENV_DIR/mmearth-bench-env
```
4. To install the needed packages in the conda environment, run
```
bash bash-scripts/install_packages.sh
```

### Generate biomass points
1. To generate the points, run
```
python generate_biomass_points.py
```
2. To merge all of the ecoregion points into a single file, run
```
python generate_biomass_points.py merge_ecoregion_points
```

### Generate soil points
1. Download the [WoSIS December 2023 snapshot](https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip)
2. Move the unzipped folder into your data directory located at `data_dir_path`
3. To generate points, run
```
python generate_soil_points.py
```

### Generate species points
1. Download the GeoJSON product for the [World Administrative Boundaries](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/)
2. Move the file into the data directory
3. Download the terrestrial mammals polygon shapefile from the [Spatial Data Download](https://www.iucnredlist.org/resources/spatial-data-download) page on the IUCN Red List
4. Move the shapefile, a folder called `MAMMALS_TERRESTRIAL_ONLY`, into `data_dir_path/species`
6. To generate points, run
```
python generate_species_points.py
```

### Generate tiles
To generate tiles, do the following for each task.

1. To save aligned modality and task data as a TIFF for every tile, run
```
python get_tile_data.py TASK
```
2. To convert the data to H5 format, run
```
python convert_to_h5.py TASK
```

### Generate splits
```
python generate_splits.py TASK
```

</details>

<details>
<summary><strong>Reproducing the results</strong></summary>

</details>

<details>
<summary><strong>Recreating the figures</strong></summary>

</details>
