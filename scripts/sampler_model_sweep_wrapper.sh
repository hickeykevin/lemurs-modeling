#!/bin/bash
# Wrapper script for Sampler x Model x Modality sweep to translate W&B args to Hydra overrides safely

ARGS=""

for arg in "$@"; do
    case "$arg" in
        sampler_pkg=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data/sampler=sweep_configs/$val\""
            ;;
        model_pkg=*)
            val="${arg#*=}"
            ARGS="$ARGS \"model=$val\""
            ;;
        modalities_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.modalities=$val\" \"data/preprocessors=$val\""
            ;;
        *)
            ARGS="$ARGS \"$arg\""
            ;;
    esac
done

eval "uv run python src/train.py \
    eval_plan=cyclical \
    task_name=cyclical_sweep \
    logger=wandb \
    data/scaler=dual \
    data.os_filter=both \
    data.collapse_strategy=none \
    data.use_demographics=true \
    data.use_sleep=false \
    data.use_survey_context=false \
    data.require_sensor_data=true \
    trainer.max_epochs=75 \
    $ARGS"
