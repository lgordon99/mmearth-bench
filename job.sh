#!/bin/bash

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem 500          # Memory pool for all cores (see also --mem-per-cpu) MBs
date
source ~/.bashrc
conda activate $1
shift
python $@
