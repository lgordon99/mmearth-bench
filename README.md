[![Website](https://img.shields.io/badge/🌍-Project_Page-black)](https://lgordon99.github.io/mmearth-bench/)

<h1 align="center">MMEarth-Bench</h1>

<p align="center">
  Lucia Gordon<sup>1,2</sup>, Serge Belongie<sup>2</sup>, Christian Igel<sup>2</sup>, Nico Lang<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;<img src="docs/static/images/harvard_logo.svg" alt="Harvard" height="80" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sup>2</sup>&nbsp;<img src="docs/static/images/ku_logo.png" alt="University of Copenhagen" height="60" style="vertical-align: middle;" />
</p>

<p align="center">
	<img src="docs/static/images/overview_figure.svg" alt="Overview figure" width="500" />
</p>

To download the MMEarth-Bench data, run
```
mkdir -p mmearth-bench-data/{biomass,soil_nitrogen,soil_organic_carbon,soil_pH,species} && for task in biomass soil_nitrogen soil_organic_carbon soil_pH species; do wget -c -P "mmearth-bench-data/$task" "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/$task/$task.h5" "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/$task/${task}_split_data.json"; done && wget -c -P mmearth-bench-data/species "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/species/species_labels.json"
```

This will create the following folder structure, occupying 59 GB.
<pre>
mmearth-bench-data/
├── biomass/
│   ├── biomass_split_data.json
│   └── biomass.h5
├── soil_nitrogen/
│   ├── soil_nitrogen_split_data.json
│   └── soil_nitrogen.h5
├── soil_organic_carbon/
│   ├── soil_organic_carbon_split_data.json
│   └── soil_organic_carbon.h5
├── soil_pH/
│   ├── soil_pH_split_data.json
│   └── soil_pH.h5
└── species/
    ├── species_labels.json
    ├── species_split_data.json
    └── species.h5
</pre>

<details>
<summary><strong>Recreating the dataset</strong></summary>

### Config file
Create a file called `config-user.yml` in the code directory with the following contents.
```
data_dir_path: 'DATA_DIR_PATH'
env_path: 'ENV_PATH'
email: 'EMAIL'
partitions: 'PARTITION_1,PARTITION_2'
gpu_partitions: 'GPU_PARTITION_1,GPU_PARTITION_2'
entity: 'WANDB_ENTITY'
project: 'WANDB_PROJECT_NAME'
```

### Setup Google Earth Engine
We use Google Earth Engine to access much of the data.

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
We use conda to manage packages for the project.

1. Open a new terminal
2. To initialize the Google Cloud CLI and create the `mmearth-bench-env` conda environment, run
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
1. Within your data directory, create `soil_nitrogen`, `soil_organic_carbon`, and `soil_pH` folders
1. Download the [WoSIS December 2023 snapshot](https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip)
2. Move `wosis_202312_nitkjd.tsv` into the `soil_nitrogen` folder, `wosis_202312_orgc.tsv` into the `soil_organic_carbon` folder, and `wosis_202312_phaq.tsv` into the `soil_pH` folder
3. To generate points, run
```
python generate_soil_points.py
```

### Generate species points
1. Move `africa.geojson` into the data directory
3. Download the terrestrial mammals polygon shapefile from the [Spatial Data Download](https://www.iucnredlist.org/resources/spatial-data-download) page on the IUCN Red List
4. Move the shapefile, a folder called `MAMMALS_TERRESTRIAL_ONLY`, into `DATA_DIR_PATH/species`
6. To generate points, run
```
python generate_species_points.py
```

### Generate datasets
To generate each task dataset, do the following for each task.

1. To save aligned modality and task data as a TIFF for every tile, run
```
python get_tile_data.py TASK
```
2. To merge the TIFFs to a single H5 file, run
```
python convert_to_h5.py TASK
```
3. To generate the train 5%, 50%, and 100%; validation; random test; and geographic test splits, run
```
python generate_splits.py TASK
```

</details>

<details>
<summary><strong>Reproducing the results</strong></summary>

The code uses [Weights & Biases](https://wandb.ai/site) to track experiments. To run the finetuning, linear probing, and joint training experiments, execute

```
bash run_FT_LP_JT.sh
```

Once these experiments are complete, you can run the test-time training experiments with
```
bash run_TTT.sh
```
</details>

<details>
<summary><strong>Recreating the figures</strong></summary>

`make_teaser_subfigures.py` ––> Figure 1

`generate_splits.py` ––> Figure 2, S.10

`view_results.py` ––> Figures 4-7, S.17-22 and Tables 5, S.13-38

`tabulate_summary_stats.py` ––> Tables S.6, S.8-S.10

`python convert_to_h5.py plot_missing_modalities` ––> Figure S.9

`generate_species_points.py` ––> Figure S.11

`python convert_to_h5.py plot_species_statistics` ––> Figure S.12-13

</details>
