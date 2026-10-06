#!/bin/bash
#SBATCH --job-name=item_deep
#SBATCH --output=logs/item_prediction/%x_%j.log
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
# Two-branch sequence model (sensor LSTM + item-history GRU) on the sensor cohort, subject-wise CV.
# Same data pipeline as run_item_sensors.sbatch (log1p + global scaling + has-data mask, 72h/6h).
# Usage: bash run_deep.sbatch <eval_plan> <name> [hydra overrides...]  (seed via seed=N)
set -eo pipefail
cd /home/khickey/lemurs-modeling
PLAN=$1; NAME=$2; shift 2
if ! timeout 15 bash -c 'echo > /dev/tcp/lemurs.wpi.edu/22' 2>/dev/null; then
  echo "gateway unreachable from $(hostname); requeueing"; sleep 30; scontrol requeue $SLURM_JOB_ID; exit 0
fi
for attempt in 1 2 3 4; do
.venv/bin/python src/train.py task_name=item_deep logger=csv trainer=cpu eval_plan=$PLAN test=true data.num_repeats=1 \
  data/sampler=sweep_configs/rolling_72h_6h_mask data/modalities=step_calorie_distance data/scaler=log1p_standard \
  data.os_filter=both data.collapse_strategy=none data.use_demographics=false \
  data.use_sleep=false data.use_survey_context=false data.require_sensor_data=true \
  model=two_branch model.auto_class_weights=true \
  trainer.max_epochs=30 trainer.enable_checkpointing=false \
  callbacks.model_checkpoint=null callbacks.early_stopping=null \
  hydra.run.dir=logs/item_prediction/runs_deep/${PLAN}_${NAME} "$@" && break
  echo "attempt $attempt failed (often the DB gateway); retrying in 90s"; sleep 90
done
