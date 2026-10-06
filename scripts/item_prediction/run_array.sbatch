#!/bin/bash
#SBATCH --job-name=item_arr
#SBATCH --output=logs/item_prediction/arr_%A_%a.log
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
# One run per manifest line ("<eval_plan> <name> [overrides...]"), throttled with --array=1-N%K so
# few jobs open the DB's SSH tunnel at once (many at once made connections time out).
# Usage: sbatch --array=1-$(wc -l < M)%4 scripts/item_prediction/run_array.sbatch <manifest>
cd /home/khickey/lemurs-modeling
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task ${SLURM_ARRAY_TASK_ID}: $LINE"
bash scripts/item_prediction/run_item.sbatch $LINE
