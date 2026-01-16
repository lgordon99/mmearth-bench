<h1 align="center">MMEarth-Bench</h1>

<p align="center">
  <img src="docs/static/images/harvard_logo.svg" alt="Harvard" height="48" align="middle" />
  &nbsp;&nbsp;
  Lucia Gordon, Serge Belongie, Christian Igel, Nico Lang
  &nbsp;&nbsp;
  <img src="docs/static/images/ku_logo.png" alt="University of Copenhagen" height="48" align="middle" />
</p>

<p align="center">
  <a href="https://lgordon99.github.io/mmearth-bench/">Project page</a>
</p>

<p align="center">
	<img src="docs/static/images/overview_figure.svg" alt="Overview figure" width="500" />
</p>

## MMEarth-Bench downstream tasks

| Task | # Tiles | Unit | Scale | Type | License |
|---|---:|---|:---:|:---:|---|
| Biomass | 18,393 | Mg/ha | Pixel-level | Regression | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| Soil Nitrogen | 5,643 | g/kg | Tile-level | Regression | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
| Soil Organic Carbon | 7,982 | g/kg | Tile-level | Regression | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
| Soil pH | 8,508 | Unitless | Tile-level | Regression | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
| Species | 36,410 | Presence/absence | Tile-level | Multi-label classification | [Terms of Use](https://www.iucnredlist.org/terms/terms-of-use) |

## MMEarth-Bench input modalities

| Modality | Bands | Scale | Type |
|---|---|:---:|:---:|
| Sentinel-2 | B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 | Pixel-level | Continuous |
| Sentinel-1 | Ascending VV, VH, HH, HV;<br>Descending VV, VH, HH, HV | Pixel-level | Continuous |
| ASTER GDEM | Elevation, slope | Pixel-level | Continuous |
| ETH Global Canopy Height | Height, uncertainty | Pixel-level | Continuous |
| Dynamic World | Landcover | Pixel-level | Categorical |
| ESA WorldCover | Landcover | Pixel-level | Categorical |
| Precipitation | Previous month, month, year | Tile-level | Continuous |
| Temperature | Previous month max, mean, min;<br>month max, mean, min;<br>year max, mean, min | Tile-level | Continuous |
| Geolocation | Longitude, latitude | Tile-level | Continuous |
| Sentinel-2 date | Date | Tile-level | Continuous |
| Biome | Biome number | Tile-level | Categorical |
| Ecoregion | Ecoregion number | Tile-level | Categorical |

## Downloading the MMEarth-Bench data

To download the MMEarth-Bench data, run
```
mkdir -p mmearth-bench-data/{biomass,soil_nitrogen,soil_organic_carbon,soil_pH,species} && for task in biomass soil_nitrogen soil_organic_carbon soil_pH species; do wget -c -P "mmearth-bench-data/$task" "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/$task/$task.h5" "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/$task/${task}_split_data.json"; done && wget -c -P mmearth-bench-data/species "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/species/species_labels.json" && wget -c -P mmearth-bench-data "https://sid.erda.dk/share_redirect/cbMhbwV1yP/mmearth-bench-data/no_data_values.json"
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
├── species/
│   ├── species_labels.json
│   ├── species_split_data.json
│   └── species.h5
└── no_data_values.json
</pre>

## Using the MMEarth-Bench data

### Reading data directly
Both input modalities and task names can be provided as keys to the H5 files. For any valid `KEY`, the following code extracts the relevant data from the H5 file at `PATH_TO_H5`.

```python
import h5py
import json

with h5py.File(PATH_TO_H5, 'r') as h5_file:
  if KEY in ['sentinel2_date', 'crs', 'sentinel2_system_index']:
    key_data = h5_file[KEY].asstr()[...]
  elif KEY in ['missing_modalities', 'species']:
    key_data = [json.loads(lst) for lst in h5_file[KEY].asstr()[...]]
  elif KEY in ['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover', 'precipitation', 'temperature',    'geolocation_encoding', 'month_encoding', 'biome', 'ecoregion', 'biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL', 'geolocation', 'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', 'SCL_NO_DATA_PIXEL_FRACTION', 'id', 'transform']:
    key_data = h5_file[KEY][:]
```

The split data JSON files can also be indexed using keys.

```python
import json

with open(PATH_TO_SPLIT_DATA_JSON, 'r') as json_file:
    split_data = json.load(json_file)

# Get indices for SPLIT ∈ ['train_100%', 'train_50%', 'train_5%', 'val', 'random_test', 'geographic_test'])
indices = split_data[f'{SPLIT}_indices']

# Get normalization statistics for MODALITY ∈ ['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'ETH_GCH', 'precipitation', 'temperature'] and SPLIT ∈ ['train_100%', 'train_50%', 'train_5%'])
means = split_data[f'{MODALITY}_{SPLIT}_means']
stds = split_data[f'{MODALITY}_{SPLIT}_stds']
```

### Using our dataset and dataloaders
Move `normalization_data.json` and `task_modalities.json` into the data directory. Then you can use `dataset.py` and `datamodule.py` to load the data with PyTorch and PyTorch Lightning.

<details>
<summary><h2 style="display: inline;">Recreating the dataset</h2></summary>

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
<summary><h2 style="display: inline;">Reproducing the results</h2></summary>

1. Move `normalization_data.json` and `task_modalities.json` into the data directory.

2. The code uses [Weights & Biases](https://wandb.ai/site) to track experiments. To run the finetuning, linear probing, and joint training experiments, execute

```
bash run_FT_LP_JT.sh
```

3. Once these experiments are complete, you can run the test-time training experiments with
```
bash run_TTT.sh
```
</details>

<details>
<summary><h2 style="display: inline;">Recreating the figures</h2></summary>

`make_teaser_subfigures.py` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figure 1

`generate_splits.py` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figure 2, S.10

`view_results.py` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figures 4-7, S.17-22 and Tables 5, S.13-38

`tabulate_summary_stats.py` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Tables S.6, S.8-S.10

`python convert_to_h5.py plot_missing_modalities` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figure S.9

`generate_species_points.py` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figure S.11

`python convert_to_h5.py plot_species_statistics` <img src="docs/static/images/right-arrow.svg" alt="→" height="12" style="vertical-align: middle;" /> Figure S.12-13

</details>

## BibTeX
```bibtex
@misc{gordon2026mmearth-bench,
  title        = {MMEarth-Bench},
  author       = {Gordon, Lucia and Belongie, Serge and Igel, Christian and Lang, Nico},
  year         = {2026},
  howpublished = {\url{https://lgordon99.github.io/mmearth-bench/}},
  note         = {Dataset and benchmark},
}
```
