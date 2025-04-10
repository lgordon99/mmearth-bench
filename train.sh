#!/bin/bash

TASK=${1}
SPLIT_TYPE=${2}
MODEL_TYPE=${3}
ADAPTATION_MODE=${4}

if [ "$TASK" == "biomass" ]; then
    MEM="60G"
elif [ "$TASK" == "species" ]; then
    MEM="30G"
else
    MEM="20G"
fi

temp_job_file=$(mktemp temp_job_XXXXXX.sh)

cat > $temp_job_file <<EOF
#!/bin/bash
#SBATCH --job-name train_${TASK}_${SPLIT_TYPE}_${MODEL_TYPE}
#SBATCH --time 1-00:00
#SBATCH --partition davies_gpu,gpu,seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:1
#SBATCH --output bash-outputs/train_${TASK}_${SPLIT_TYPE}_${MODEL_TYPE}.out

source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
echo "Task: ${TASK}"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
python train.py +task=${TASK} +split_type=${SPLIT_TYPE} +model_type=${MODEL_TYPE} +adaptation_mode=${ADAPTATION_MODE}

EOF

sbatch $temp_job_file
rm $temp_job_file
