#!/bin/bash

#SBATCH -p seas_compute
#SBATCH -t 24:00:00
#SBATCH --mem 130G
#SBATCH -o upload_to_erda.out

date
source ~/.bashrc

# Use sftp with a here-document to pass commands
sftp erda <<EOF
put /n/gajos_lab/Lab/luciagordon/mmearth-bench/biomass/biomass.h5 mmearth-bench/mmearth-bench-data/biomass
put /n/gajos_lab/Lab/luciagordon/mmearth-bench/soil_nitrogen/soil_nitrogen.h5 mmearth-bench/mmearth-bench-data/soil_nitrogen
put /n/gajos_lab/Lab/luciagordon/mmearth-bench/soil_organic_carbon/soil_organic_carbon.h5 mmearth-bench/mmearth-bench-data/soil_organic_carbon
put /n/gajos_lab/Lab/luciagordon/mmearth-bench/soil_pH/soil_pH.h5 mmearth-bench/mmearth-bench-data/soil_pH
put /n/gajos_lab/Lab/luciagordon/mmearth-bench/species/species.h5 mmearth-bench/mmearth-bench-data/species
quit
EOF

date
