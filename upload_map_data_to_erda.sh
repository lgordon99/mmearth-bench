#!/bin/bash

#SBATCH -p seas_compute
#SBATCH -t 36:00:00
#SBATCH --mem 5G
#SBATCH --account davies_lab

date
source ~/.bashrc

TASK="${1:-}"
echo "Subfolder: $TASK"

# Use sftp with a here-document to pass commands
sftp erda <<EOF
-mkdir mmearth-bench/mmearth-bench-explorer
put -r /n/gajos_lab/Lab/luciagordon/mmearth-bench/mmearth-bench-explorer/$TASK mmearth-bench/mmearth-bench-explorer/$TASK
quit
EOF

date
