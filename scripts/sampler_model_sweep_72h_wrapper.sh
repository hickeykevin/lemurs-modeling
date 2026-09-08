#!/bin/bash
# Wrapper script for Sampler x Model x Modality sweep (72h purge) to translate W&B args to Hydra overrides safely

CMD_ARGS=(
    eval_plan=cyclical
    task_name=cyclical_sweep_72h
    logger=wandb
    data/scaler=dual
    data.os_filter=both
    data.collapse_strategy=none
    data.use_demographics=true
    data.use_sleep=false
    data.use_survey_context=false
    data.require_sensor_data=true
    data.purge_hours=72.0
    trainer.max_epochs=75
)

for arg in "$@"; do
    case "$arg" in
        sampler_pkg=*)
            val="${arg#*=}"
            CMD_ARGS+=("data/sampler=sweep_configs/$val")
            ;;
        model_pkg=*)
            val="${arg#*=}"
            CMD_ARGS+=("model=$val")
            ;;
        modalities_choice=*)
            val="${arg#*=}"
            CMD_ARGS+=("data.modalities=$val" "data/preprocessors=$val")
            ;;
        *)
            CMD_ARGS+=("$arg")
            ;;
    esac
done

exec uv run python src/train.py "${CMD_ARGS[@]}"
