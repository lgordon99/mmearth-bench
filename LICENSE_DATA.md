# MMEarth-Bench Licensing & Terms of Use

The MMEarth-Bench benchmark is a curated collection of data aggregated from multiple source datasets.

## 1. Summary of Task Dataset Licenses

Each task dataset within this benchmark is governed by its original source license:

| Task | Source Dataset | License | Commercial Use Allowed? |
| :--- | :--- | :--- | :--- |
| **Biomass** | GEDI L4A | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |
| **Soil Nitrogen** | WoSIS December 2023 Snapshot | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | No |
| **Soil Organic Carbon**| WoSIS December 2023 Snapshot | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | No |
| **Soil pH** | WoSIS December 2023 Snapshot | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | No |
| **Species** | IUCN Red List Terrestrial Mammal Ranges | [IUCN Terms and Conditions of Use](https://www.iucnredlist.org/terms/terms-of-use) | No |

\*With attribution

---

## 2. Summary of Modality Licenses

Each input modality is governed by its original source license:

| Modality | Source Dataset | License | Commercial Use Allowed? |
| :--- | :--- | :--- | :--- |
| **Sentinel-2** | Copernicus Sentinel-2 L2A (SR Harmonized) | [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) | Yes\* |
| **Sentinel-1** | Copernicus Sentinel-1 GRD | [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) | Yes\* |
| **ASTER GDEM** | ASTER Global Digital Elevation Model V3 | No restrictions (similar to [CC0](https://creativecommons.org/publicdomain/zero/1.0/)) | Yes\* |
| **ETH Global Canopy Height** | ETH Global Canopy Height 2020 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |
| **Dynamic World** | Dynamic World V1 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |
| **ESA WorldCover** | ESA WorldCover 10m v100 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |
| **Precipitation** | ERA5-Land Monthly Aggregated | [Copernicus C3S/CAMS License](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) | Yes\* |
| **Temperature** | ERA5-Land Monthly Aggregated | [Copernicus C3S/CAMS License](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) | Yes\* |
| **Geolocation** | Derived from tile geometry | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (benchmark curation) | Yes\* |
| **Sentinel-2 date** | Derived from Sentinel-2 metadata | [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) | Yes\* |
| **Biome** | RESOLVE Ecoregions 2017 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |
| **Ecoregion** | RESOLVE Ecoregions 2017 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes\* |

\*With attribution

## 3. Overall Collection Terms

* **Commercial Use Notice:** Because the soil and species tasks are distributed under non-commercial licenses, the overall benchmark **as a complete package** cannot be used for commercial purposes. Users seeking commercial evaluation should restrict their usage strictly to the biomass task (and may use the modalities under their respective licenses above).
* **Benchmark Curation & Metadata:** The JSON files containing the split data and no data values created by the authors are released under the CC BY License.

---

## 4. Individual Task Dataset Attributions & Licenses

### Task 1: Biomass
* **Creator:** [The Global Ecosystem Dynamics Investigation: High-resolution laser ranging of the Earth’s forests and topography](https://www.sciencedirect.com/science/article/pii/S2666017220300018)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** [GEDI L4A Aboveground Biomass Density, Version 2.1](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI04_A_002)

---

### Task 2–4: Soil Nitrogen, Organic Carbon, & pH
* **Creator:** [Providing quality-assessed and standardised soil data to support global mapping and modelling (WoSIS snapshot 2023)](https://essd.copernicus.org/articles/16/4735/2024/)
* **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
* **Source:** [WoSIS snapshot - December 2023](https://data.isric.org/geonetwork/srv/api/records/e50f84e1-aa5b-49cb-bd6b-cd581232a2ec)

---

### Task 5: Species
* **Creator:** [IUCN Red List of Threatened Species Version 2025-1](https://www.iucnredlist.org)
* **License:** [The IUCN Red List Terms and Conditions of Use (version 3.1, June 2024*)](https://www.iucnredlist.org/terms/terms-of-use)
* **Source:** [Spatial Data Download](https://www.iucnredlist.org/resources/spatial-data-download)

---

## 5. Individual Modality Attributions & Licenses

### Sentinel-2
* **Creator:** European Space Agency / Copernicus Programme
* **License:** [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) (free, full, and open access; attribution required)
* **Source:** [COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
* **Attribution:** Contains modified Copernicus Sentinel data 2020

### Sentinel-1
* **Creator:** European Space Agency / Copernicus Programme
* **License:** [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) (free, full, and open access; attribution required)
* **Source:** [COPERNICUS/S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
* **Attribution:** Contains modified Copernicus Sentinel data 2020

### ASTER GDEM
* **Creator:** NASA / METI
* **License:** No restrictions on reuse, sale, or redistribution; citation requested
* **Source:** [ASTER Global Digital Elevation Model V3](https://lpdaac.usgs.gov/products/astgtmv003/) (Earth Engine: `projects/sat-io/open-datasets/ASTER/GDEM`)
* **Attribution:** ASTER GDEM is a product of METI and NASA

### ETH Global Canopy Height
* **Creator:** [A high-resolution canopy height model of the Earth](https://www.nature.com/articles/s41559-023-02206-6)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** [ETH Global Canopy Height 2020](https://gee-community-catalog.org/projects/canopy/) (Earth Engine: `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`)

### Dynamic World
* **Creator:** [Dynamic World, Near real-time global 10 m land use land cover mapping](https://www.nature.com/articles/s41597-022-01307-4)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** [GOOGLE/DYNAMICWORLD/V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)
* **Attribution:** This dataset is produced for the Dynamic World Project by Google in partnership with National Geographic Society and the World Resources Institute. Contains modified Copernicus Sentinel data [2015-present].

### ESA WorldCover
* **Creator:** [ESA WorldCover 10 m 2020 v100](https://doi.org/10.5281/zenodo.5571936)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** [ESA/WorldCover/v100](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100)

### Precipitation & Temperature
* **Creator:** [ERA5-Land monthly averaged data from 1981 to present](https://doi.org/10.24381/cds.68d2bb30)
* **License:** [Copernicus C3S/CAMS License](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products)
* **Source:** [ECMWF/ERA5_LAND/MONTHLY_AGGR](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR)
* **Attribution:** Generated using Copernicus Climate Change Service Information 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus Information or Data it contains.

### Geolocation
* **Creator:** MMEarth-Bench (derived from tile geometry)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** Longitude/latitude corresponding to each tile center

### Sentinel-2 date
* **Creator:** European Space Agency / Copernicus Programme
* **License:** [Copernicus Sentinel Data License](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice)
* **Source:** Sentinel-2 acquisition date associated with each tile

### Biome & Ecoregion
* **Creator:** [An Ecoregion-Based Approach to Protecting Half the Terrestrial Realm](https://doi.org/10.1093/biosci/bix014)
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Source:** [RESOLVE/ECOREGIONS/2017](https://developers.google.com/earth-engine/datasets/catalog/RESOLVE_ECOREGIONS_2017)
