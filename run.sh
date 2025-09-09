task=${1}
architecture=${2}
adaptation_mode=${3}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")

if [ "$TASK" == "biomass" ]; then
    MEM="70G"
elif [ "$TASK" == "species" ]; then
    MEM="130G"
else
    MEM="30G"
fi

bash_file="${data_dir_path}/experiments/${task}_${architecture}_${adaptation_mode}_run.sh"

cat > $bash_file <<EOF
#!/bin/bash
#SBATCH --job-name ${task}_${architecture}_${adaptation_mode}
#SBATCH --time 1-00:00
#SBATCH --partition gpu,seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output $data_dir_path/experiments/output-files/$task_$architecture_$adaptation_mode.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate $data_dir_path/mmearth-bench-env
echo "Task: ${task}"
echo "Architecture: ${architecture}"
echo "Adaptation Mode: ${adaptation_mode}"
python train.py +task=${task} +architecture=${architecture} +adaptation_mode=${adaptation_mode}
EOF

sbatch $bash_file
rm $bash_file
