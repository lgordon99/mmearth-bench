task=${1}
model_type=${2}
adaptation_mode=${3}

sweep_log_file="sweep_log.csv"

if [ ! -f "$sweep_log_file" ]; then
    echo "date,name,sweep_ID" > "$sweep_log_file"
fi

sweep_name="${task}_${model_type}_${adaptation_mode}_bayes"

touch sweep_2.yaml

yaml_content="project: mmearth-bench
entity: luciagordon-harvard-university
name: ${sweep_name}
program: train.py
method: bayes
metric:
  name: \"Val RMSE\"
  goal: minimize
parameters:
  model.max_lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 1.0
  model.weight_decay:
    distribution: log_uniform_values
    min: 0.00001
    max: 1.0
run_cap: 30
command:
  - python
  - train.py
  - +task=${task}
  - +model_type=${model_type}
  - +adaptation_mode=${adaptation_mode}"

echo "$yaml_content" > sweep_2.yaml

wandb sweep sweep_2.yaml &> sweep_output.txt
rm sweep_2.yaml
sweep_id=$(cat sweep_output.txt | grep "agent" | tail -1 | awk '{print $NF}')
rm sweep_output.txt

echo "Sweep ID: $sweep_id"

if [ -n "$sweep_id" ]; then
    current_date=$(date "+%Y-%m-%d %H:%M:%S")
    echo "$current_date,$sweep_name,$sweep_id" >> "$sweep_log_file"
fi

if [ "$task" == "biomass" ]; then
    MEM="60G"
elif [ "$task" == "species" ]; then
    MEM="30G"
else
    MEM="20G"
fi

touch spawn_agent.sh

cat > spawn_agent.sh <<EOF
#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time 0-40:00
#SBATCH --partition gpu,seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output bash-outputs/sweep_%j.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate /n/davies_lab/Users/luciagordon/mmearth-bench/mmearth-bench-env
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
wandb agent $sweep_id
EOF

sbatch spawn_agent.sh
rm spawn_agent.sh
