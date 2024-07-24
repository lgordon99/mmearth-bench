#!/bin/bash

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --mem 500          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
date
source ~/.bashrc
conda activate $1
shift
python $@
