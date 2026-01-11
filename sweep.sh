task=${1}
architecture=${2}
adaptation_mode=${3}
tuning_mode=${4}

data_dir_path=$(python -c "import yaml; print(yaml.safe_load(open('config-user.yml'))['data_dir_path'])")
sweep_hyperparameters="${data_dir_path}/experiments/sweep_hyperparameters.json"

max_lr=($(jq -r ".[\"${tuning_mode}\"].max_lr | .[]" $sweep_hyperparameters))
echo "max_lr: ${max_lr[@]}"

weight_decay=($(jq -r ".[\"${tuning_mode}\"].weight_decay | .[]" $sweep_hyperparameters))
echo "weight_decay: ${weight_decay[@]}"

if [ "$tuning_mode" == "llrd" ]; then
  decay_factor=($(jq -r ".[\"${tuning_mode}\"].decay_factor | .[]" $sweep_hyperparameters))
  echo "decay_factor: ${decay_factor[@]}"
fi

if [ "$tuning_mode" == "llrd" ]; then
  num_runs=$(( ${#max_lr[@]} * ${#weight_decay[@]} * ${#decay_factor[@]} ))
else
  num_runs=$(( ${#max_lr[@]} * ${#weight_decay[@]} ))
fi

function format_array_to_string() {
    local -n array=$1
    local result="["

    for i in "${!array[@]}"; do
        if [ "$i" -eq 0 ]; then
            result+="${array[$i]}"
        else
            result+=", ${array[$i]}"
        fi
    done

    result+="]"
    echo "$result"
}

max_lr_string=$(format_array_to_string max_lr)
weight_decay_string=$(format_array_to_string weight_decay)

if [ "$tuning_mode" == "llrd" ]; then
    decay_factor_string=$(format_array_to_string decay_factor)
fi

sweep_log_file="${data_dir_path}/experiments/sweep_log.csv"

if [ ! -f "$sweep_log_file" ]; then
    echo "date,name,sweep_ID" > "$sweep_log_file"
fi

sweep_name="${task}_${architecture}_${adaptation_mode}_${tuning_mode}"

yaml_content="project: mmearth-bench
entity: luciagordon-harvard-university
name: ${sweep_name}
program: train.py
method: grid
metric:
  name: \"Val RMSE\"
  goal: minimize
parameters:
  model.max_lr:
    values: ${max_lr_string}
  model.weight_decay:
    values: ${weight_decay_string}"

if [ "$tuning_mode" == "llrd" ]; then
  yaml_content="${yaml_content}
  model.decay_factor:
    values: ${decay_factor_string}"
fi

yaml_content="${yaml_content}
command:
  - python
  - train.py
  - +task=${task}
  - +architecture=${architecture}
  - +adaptation_mode=${adaptation_mode}
  - model.tuning_mode=${tuning_mode}"

sweep_yaml="${data_dir_path}/experiments/sweep.yaml"
echo "$yaml_content" > "$sweep_yaml"
sweep_output="${data_dir_path}/experiments/sweep_output.txt"
wandb sweep $sweep_yaml &> $sweep_output
rm $sweep_yaml
sweep_id=$(cat $sweep_output | grep "agent" | tail -1 | awk '{print $NF}')
rm $sweep_output

echo "Sweep ID: $sweep_id"

if [ -z "$sweep_id" ]; then
    echo "Error: sweep_id is empty. Exiting."
    exit 1
fi

echo "Number of runs: $num_runs"

if [ -n "$sweep_id" ]; then
    current_date=$(date "+%Y-%m-%d %H:%M:%S")
    echo "$current_date,$sweep_name,$sweep_id" >> "$sweep_log_file"
fi

if [ "$task" == "biomass" ]; then
    HOURS="30"
else
    HOURS="10"
fi

if [ "$task" == "biomass" ]; then
    MEM="70G"
elif [ "$task" == "species" ]; then
    MEM="30G"
else
    MEM="30G"
fi

spawn_agent_bash="${data_dir_path}/experiments/spawn_agent.sh"

cat > $spawn_agent_bash <<EOF
#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time $HOURS:00:00
#SBATCH --partition gpu,seas_gpu
#SBATCH --mem $MEM
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output $data_dir_path/experiments/output-files/sweep_%j.out
#SBATCH --account davies_lab

source ~/.bashrc
conda activate /n/gajos_lab/Lab/luciagordon/mmearth-bench/mmearth-bench-env
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: \$NUM_GPUS"
wandb agent --count 1 $sweep_id
EOF

for _ in $(seq 1 $num_runs); do
    sbatch $spawn_agent_bash
done

rm $spawn_agent_bash
