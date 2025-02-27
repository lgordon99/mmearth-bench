#!/bin/bash

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine

date
source ~/.bashrc
conda activate $1
shift
python $@
