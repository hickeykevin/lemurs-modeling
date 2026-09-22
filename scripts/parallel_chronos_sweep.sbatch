#!/bin/bash
#SBATCH --job-name=chronos_sweep
#SBATCH --output=logs/chronos_sweep_%j.log
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00

if [ -z "$1" ]; then
    echo "Error: No sweep ID provided."
    echo "Usage: sbatch scripts/parallel_chronos_sweep.sbatch <sweep_id> [num_workers]"
    exit 1
fi

SWEEP_ID=$1
# Auto-prepend entity/project if user passed a bare sweep ID (e.g. 3sle2zr6)
if [[ "$SWEEP_ID" != */* ]]; then
    SWEEP_ID="hickeykevin/lemurs-modeling/$SWEEP_ID"
fi

# Default to 8 parallel workers on 32 CPUs (giving each PyTorch worker 4 CPU threads)
NUM_WORKERS=${2:-8}
THREADS_PER_WORKER=$(( 32 / NUM_WORKERS ))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then
    THREADS_PER_WORKER=1
fi

# Set intra-op threading for PyTorch / MKL tensor operations
export OMP_NUM_THREADS=$THREADS_PER_WORKER
export MKL_NUM_THREADS=$THREADS_PER_WORKER
export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_WORKER
export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER

mkdir -p logs

echo "Allocating ${SLURM_CPUS_PER_TASK:-32} CPUs: launching $NUM_WORKERS workers with $THREADS_PER_WORKER threads each for sweep $SWEEP_ID..."

for i in $(seq 1 $NUM_WORKERS); do
    uv run wandb agent $SWEEP_ID > logs/chronos_agent_${SLURM_JOB_ID}_worker_${i}.log 2>&1 &
done

wait
echo "All parallel Chronos-Bolt agents have finished."
