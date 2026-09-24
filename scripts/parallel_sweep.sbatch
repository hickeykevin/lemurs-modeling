#!/bin/bash
#SBATCH --job-name=parallel_sweep
#SBATCH --output=logs/parallel_sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=short

if [ -z "$1" ]; then
    echo "Error: No sweep ID provided."
    echo "Usage: sbatch scripts/parallel_sweep.sbatch <sweep_id> [num_workers]"
    exit 1
fi

SWEEP_ID=$1
# Auto-prepend entity/project if user passed a bare sweep ID (e.g. 3sle2zr6)
if [[ "$SWEEP_ID" != */* ]]; then
    SWEEP_ID="hickeykevin/lemurs-modeling/$SWEEP_ID"
fi

# Allow setting NUM_WORKERS dynamically (defaults to 16 for tabular; 8 recommended for DL)
NUM_WORKERS=${2:-16}
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-32}
THREADS_PER_WORKER=$(( TOTAL_CPUS / NUM_WORKERS ))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then
    THREADS_PER_WORKER=1
fi

# Set intra-op threading for PyTorch / numpy / MKL operations
export OMP_NUM_THREADS=$THREADS_PER_WORKER
export MKL_NUM_THREADS=$THREADS_PER_WORKER
export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_WORKER
export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER

mkdir -p logs

echo "Allocating $TOTAL_CPUS CPUs: launching $NUM_WORKERS W&B Agents ($THREADS_PER_WORKER threads each) for sweep: $SWEEP_ID..."

for i in $(seq 1 $NUM_WORKERS); do
    uv run wandb agent $SWEEP_ID > logs/agent_${SLURM_JOB_ID}_worker_${i}.log 2>&1 &
done

wait
echo "All parallel agents have finished."
