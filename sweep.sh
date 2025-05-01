NUM_RUNS=${1}
wandb sweep sweep.yaml &> sweep_output.txt
SWEEP_ID=$(cat sweep_output.txt | grep "agent" | tail -1 | awk '{print $NF}')
rm sweep_output.txt

echo "Sweep ID: $SWEEP_ID"
echo "Number of runs: $NUM_RUNS"

for _ in $(seq 1 $NUM_RUNS); do
    sbatch spawn_agent.sh $SWEEP_ID
done
