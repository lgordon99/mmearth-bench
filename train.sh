#!/bin/bash

TASK=${1}
SPLIT_TYPE=${2}
MODEL_TYPE=${3}

if [ "$TASK" == "biomass" ]; then
    MEM="60G"
else
    MEM="20G"
fi

cat > temp_job.sh <<EOF
#!/bin/bash
#SBATCH --job-name run_model
#SBATCH --time 0-00:10
#SBATCH --partition davies_gpu,gpu,seas_gpu
#SBATCH --mem ${MEM}
#SBATCH --gres gpu:1
#SBATCH --output bash-outputs/train_${TASK}_${SPLIT_TYPE}_${MODEL_TYPE}.out

source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
echo "Task: ${TASK}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
SCRATCH_DIR=/scratch
echo "Copying h5 file to \$SCRATCH_DIR"
H5_FILE=/n/davies_lab/Users/luciagordon/mmearth-bench/${TASK}/${TASK}_h5.hdf5
cp \$H5_FILE \$SCRATCH_DIR
echo "Copied h5 file to \$SCRATCH_DIR"
SPLIT_DATA=/n/davies_lab/Users/luciagordon/mmearth-bench/${TASK}/${TASK}_${SPLIT_TYPE}_split_data.json
if [ -e ${SPLIT_DATA} ]; then
    echo "Copying split data to \$SCRATCH_DIR"
    cp \$SPLIT_DATA \$SCRATCH_DIR
    echo "Copied split data to \$SCRATCH_DIR"
fi
python train.py +task=${TASK} +split_type=${SPLIT_TYPE} +model_type=${MODEL_TYPE}

EOF

sbatch temp_job.sh
rm temp_job.sh
