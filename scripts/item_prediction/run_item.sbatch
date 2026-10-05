#!/bin/bash
#SBATCH --job-name=item_pred
#SBATCH --output=logs/item_prediction/%x_%j.log
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
# Item-history prediction: raw item scores of one theme on the previous K surveys -> suicide_risk
# at survey t. No sensor features (use_sequence_features=false); the sampler/modality are only
# needed to build the cohort. Subject-wise (user_cv) or within-person (cyclical) evaluation.
# Usage: sbatch --job-name=<name> logs/item_prediction/run_item.sbatch <eval_plan> <name> [hydra overrides...]
set -eo pipefail
cd /home/khickey/lemurs-modeling
PLAN=$1; NAME=$2; shift 2
# Some compute nodes cannot reach the DB's SSH gateway (seen 2026-10-05); retry on another node.
if ! timeout 15 bash -c 'echo > /dev/tcp/lemurs.wpi.edu/22' 2>/dev/null; then
  echo "gateway unreachable from $(hostname); requeueing"; sleep 30; scontrol requeue $SLURM_JOB_ID; exit 0
fi
EXTRA=""
if [ "$PLAN" = "user_cv" ]; then EXTRA="test=true data.num_repeats=1"; else EXTRA="data.purge_hours=120.0"; fi
for attempt in 1 2 3 4; do
.venv/bin/python src/train.py task_name=item_prediction logger=csv trainer=cpu eval_plan=$PLAN $EXTRA \
  data/sampler=sweep_configs/rolling_72h_6h data/modalities=step data/scaler=dual \
  data.os_filter=both data.collapse_strategy=none data.use_demographics=false \
  data.use_sleep=false data.use_survey_context=false data.require_sensor_data=false \
  model=flaml_xgboost model.automl_config.metric=roc_auc model.auto_class_weights=true \
  model.use_sequence_features=false trainer.max_epochs=75 trainer.enable_checkpointing=false \
  callbacks.model_checkpoint=null callbacks.early_stopping=null \
  hydra.run.dir=logs/item_prediction/runs/${PLAN}_${NAME} "$@" && break
  echo "attempt $attempt failed (often the DB gateway); retrying in 90s"; sleep 90
done
