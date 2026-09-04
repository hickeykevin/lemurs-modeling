#!/bin/bash
#SBATCH --job-name=parallel_sweep
#SBATCH --output=logs/parallel_sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00

if [ -z "$1" ]; then
    echo "Error: No sweep ID provided."
    echo "Usage: sbatch scripts/parallel_sweep.sbatch <sweep_id>"
    exit 1
fi

SWEEP_ID=$1
# Auto-prepend entity/project if user passed a bare sweep ID (e.g. 3sle2zr6)
if [[ "$SWEEP_ID" != */* ]]; then
    SWEEP_ID="hickeykevin/lemurs-modeling/$SWEEP_ID"
fi
NUM_WORKERS=16

# Prevent CPU core thrashing by restricting each PyTorch/numpy thread count to 1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Starting $NUM_WORKERS W&B Agents in parallel for sweep: $SWEEP_ID..."

for i in $(seq 1 $NUM_WORKERS); do
    uv run wandb agent $SWEEP_ID > logs/agent_${SLURM_JOB_ID}_worker_${i}.log 2>&1 &
done

wait
echo "All parallel agents have finished."
