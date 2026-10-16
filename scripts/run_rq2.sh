#!/bin/bash
# Runs the step + distance grid behind the paper's RQ2 table and RQ3 heatmap: each pipeline
# (a table row) on each of the 20 lookback x bin samplers. Every run uses the fixed protocol
# in configs/ (cyclical walk-forward CV, dual scaler, hours slept, 120 h purge, pinned
# database rows) and writes its metrics to logs/<task_name>/runs/<time>/.
#
# Usage:
#   bash scripts/run_rq2.sh                          # all 20 pipelines x 20 samplers (400 runs)
#   bash scripts/run_rq2.sh xgboost_stats tirex      # chosen pipelines, all samplers
#   SAMPLERS="72h_6h" bash scripts/run_rq2.sh xgboost_stats
#   EXTRA="logger=wandb" bash scripts/run_rq2.sh     # any extra Hydra overrides
#   FM_TRAINER=cpu bash scripts/run_rq2.sh tirex     # the paper ran the four encoders on a GPU
#
# Pipelines:
#   lr svm rf xgboost                                tabular models on the raw binned window
#   lr_stats svm_stats rf_stats xgboost_stats        ... on summary statistics
#   lr_catch22 svm_catch22 rf_catch22 xgboost_catch22  ... on catch22 features
#   lstm transformer                                 sequence models
#   tabpfn tabpfn_ft                                 TabPFN-3.5 frozen / fine-tuned (extra: tabpfn)
#   tirex tirex_ft                                   TiRex frozen / fine-tuned (extra: tirex)
#   chronos_bolt chronos_bolt_ft                     Chronos-Bolt frozen / fine-tuned (extra: chronos)
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_PIPELINES="lr svm rf xgboost lr_stats svm_stats rf_stats xgboost_stats
  lr_catch22 svm_catch22 rf_catch22 xgboost_catch22 lstm transformer
  tabpfn tabpfn_ft tirex tirex_ft chronos_bolt chronos_bolt_ft"
ALL_SAMPLERS="24h_4h 24h_6h 24h_8h 24h_12h 48h_4h 48h_6h 48h_8h 48h_12h
  72h_4h 72h_6h 72h_8h 72h_12h 96h_4h 96h_6h 96h_8h 96h_12h 120h_4h 120h_6h 120h_8h 120h_12h"
FM_TRAINER=${FM_TRAINER:-gpu}

# Hydra overrides for one pipeline
pipeline_args() {
    case "$1" in
        lr|svm|rf|xgboost)                         echo "model=flaml_$1" ;;
        *_stats)                                   echo "model=flaml_${1%_stats} model/featurizer=summary_stats" ;;
        *_catch22)                                 echo "model=flaml_${1%_catch22} model/featurizer=catch22" ;;
        lstm|transformer)                          echo "model=$1" ;;
        tabpfn|tabpfn_ft)                          echo "model=flaml_$1" ;;
        tirex|chronos_bolt)                        echo "model=$1 trainer=$FM_TRAINER" ;;
        tirex_ft|chronos_bolt_ft)                  echo "model=$1 trainer=$FM_TRAINER trainer.max_epochs=15" ;;
        *) echo "unknown pipeline: $1" >&2; return 1 ;;
    esac
}

PIPELINES=${*:-$ALL_PIPELINES}
for p in $PIPELINES; do
    args=$(pipeline_args "$p")
    for s in ${SAMPLERS:-$ALL_SAMPLERS}; do
        echo ">>> $p $s"
        # shellcheck disable=SC2086
        uv run python src/train.py $args data/sampler=sweep_configs/rolling_$s \
            task_name="rq2_${p}_${s}" ${EXTRA:-}
    done
done
