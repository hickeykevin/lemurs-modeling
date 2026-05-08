#!/bin/bash
#SBATCH --job-name=health_sweep
#SBATCH --output=logs/sweep_%A_%a.log
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=1-40

# Usage: sbatch scripts/run_sweep.sbatch <sweep_id>

if [ -z "$1" ]; then
    echo "Error: No sweep ID provided."
    echo "Usage: sbatch scripts/run_sweep.sbatch <sweep_id>"
    exit 1
fi

SWEEP_ID=$1

echo "Starting W&B Agent for sweep: $SWEEP_ID (Task $SLURM_ARRAY_TASK_ID)"

# Run the agent using uv
uv run wandb agent $SWEEP_ID
