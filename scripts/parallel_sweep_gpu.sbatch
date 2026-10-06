#!/bin/bash
#SBATCH --job-name=fm_sweep
#SBATCH --output=logs/fm_sweep_%j.log
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Runs N W&B sweep agents that share one GPU (foundation-model sweeps).
# Usage: sbatch scripts/parallel_sweep_gpu.sbatch <sweep_id> [num_agents]
set -eo pipefail
cd /home/khickey/lemurs-modeling
[ -z "$1" ] && { echo "usage: sbatch scripts/parallel_sweep_gpu.sbatch <sweep_id> [num_agents]"; exit 1; }
SWEEP_ID=$1; [[ "$SWEEP_ID" != */* ]] && SWEEP_ID="hickeykevin/lemurs-modeling/$SWEEP_ID"
N=${2:-3}
T=$(( ${SLURM_CPUS_PER_TASK:-16} / N )); [ "$T" -lt 1 ] && T=1
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) | $N agents x $T threads | sweep $SWEEP_ID"
for i in $(seq 1 $N); do
  uv run wandb agent "$SWEEP_ID" > logs/fm_agent_${SLURM_JOB_ID}_${i}.log 2>&1 &
  sleep 30   # stagger: agents opening the DB tunnel at once has timed out before
done
wait
echo "all agents finished $(date)"
