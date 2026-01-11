#!/bin/bash
HYDRA_FULL_ERROR=1
TASK=${1}
MODEL_TYPE=${2}
ADAPTATION_MODE=${3}

if [ "$#" -gt 3 ]; then
    MAX_LR=${4}
    WEIGHT_DECAY=${5}
    DECAY_FACTOR=${6}
fi

if [ "$TASK" == "biomass" ]; then
    MEM="70G"
elif [ "$TASK" == "species" ]; then
    MEM="30G"
else
    MEM="30G"
fi

temp_job_file=$(mktemp temp_job_XXXXXX.sh)
touch $temp_job_file

# job_file_content="#!/bin/bash
# #SBATCH --job-name train_${TASK}_${MODEL_TYPE}
# #SBATCH --time 03:00:00
# #SBATCH --partition gpu,seas_gpu
# #SBATCH --mem $MEM
# #SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --output bash-outputs/train_${TASK}_${MODEL_TYPE}.out

# source ~/.bashrc
# conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
# echo \"Task: ${TASK}\"
# NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# echo \"Number of GPUs: \$NUM_GPUS\""

# if [ "$#" -ne 3 ]; then
#     MAX_LR=${4}
#     WEIGHT_DECAY=${5}
# fi

cat > $temp_job_file <<EOF
#!/bin/bash
#SBATCH --job-name train_${TASK}_${MODEL_TYPE}_${ADAPTATION_MODE}
#SBATCH --time 2-00:00
#SBATCH --partition seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output bash-outputs/train_${TASK}_${MODEL_TYPE}_${ADAPTATION_MODE}_%j.out
#SBATCH --account gajos_lab

source ~/.bashrc
conda activate /n/gajos_lab/Lab/luciagordon/mmearth-bench/mmearth-bench-env
echo "Task: ${TASK}"
# NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# echo "Number of GPUs: \$NUM_GPUS"
# python train.py +task=${TASK} +model_type=${MODEL_TYPE} +adaptation_mode=${ADAPTATION_MODE} model.max_lr=${MAX_LR} model.weight_decay=${WEIGHT_DECAY}
# python train.py +task=${TASK} +architecture=${MODEL_TYPE} +adaptation_mode=${ADAPTATION_MODE} datamodule.batch_size=1
python train.py +task=${TASK} +architecture=${MODEL_TYPE} +adaptation_mode=${ADAPTATION_MODE}

EOF

sbatch $temp_job_file
rm $temp_job_file
# python train.py +task=${TASK} +model_type=${MODEL_TYPE} +adaptation_mode=${ADAPTATION_MODE} model.max_lr=${MAX_LR} model.weight_decay=${WEIGHT_DECAY} model.decay_factor=${DECAY_FACTOR}
