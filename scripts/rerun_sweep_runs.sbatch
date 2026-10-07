#!/bin/bash
#SBATCH --job-name=sweep_rerun
#SBATCH --output=logs/sweep_checks/rerun_%A_%a.log
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Re-runs individual sweep combinations outside the W&B agent (a grid sweep never re-issues a
# combination it already dispatched), one run per GPU. Each manifest line is
#   <sweep name> [timeout=<duration>] key=value ...
# and is replayed through the sweep's own wrapper with the sweep YAML's fixed command, so the
# run gets the same settings, W&B group and run name as a sweep run.
# Usage: LEMURS_DB_CACHE_DIR=<snapshot> sbatch --array=1-N%K scripts/rerun_sweep_runs.sbatch <manifest>
set -eo pipefail
cd /home/khickey/lemurs-modeling
[ -d "${LEMURS_DB_CACHE_DIR:-/nonexistent}" ] || { echo "LEMURS_DB_CACHE_DIR must point at the database snapshot"; exit 1; }
export LEMURS_DB_CACHE_DIR
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
read -r SWEEP REST <<< "$LINE"
LIMIT=""
if [[ "$REST" == timeout=* ]]; then LIMIT="${REST%% *}"; LIMIT="${LIMIT#timeout=}"; REST="${REST#* }"; fi
mapfile -t CMD < <(.venv/bin/python -c "import yaml,sys; c=yaml.safe_load(open('configs/hparams_search/$SWEEP.yaml'))['command']; print('\n'.join(x for x in c if x != '\${args_no_hyphens}'))")
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) | task ${SLURM_ARRAY_TASK_ID}: $SWEEP $REST ${LIMIT:+(limit $LIMIT)}"
read -r -a PARAMS <<< "$REST"
if [ -n "$LIMIT" ]; then
  timeout --signal=TERM --kill-after=60 "$LIMIT" "${CMD[@]}" "${PARAMS[@]}" || { rc=$?; echo "exit code $rc (124 = hit the $LIMIT limit)"; exit $rc; }
else
  "${CMD[@]}" "${PARAMS[@]}"
fi
echo "done $(date)"
