#!/bin/bash
# Wrapper script for RQ1 FLAML Models sweep
# Translates W&B sweep parameters into Hydra config overrides and attaches W&B logging metadata

set -eo pipefail

ARGS=()
SAMPLER_VAL=""
MODEL_VAL=""
MODALITIES_VAL=""
HAS_UV_PREFIX=false

# Check if command line starts with 'uv' (as configured in RQ1_flaml_models.yaml)
if [ "$1" = "uv" ]; then
    HAS_UV_PREFIX=true
fi

for arg in "$@"; do
    case "$arg" in
        sampler_pkg=*)
            SAMPLER_VAL="${arg#*=}"
            ARGS+=("data/sampler=sweep_configs/$SAMPLER_VAL")
            ;;
        model_pkg=*)
            MODEL_VAL="${arg#*=}"
            ARGS+=("model=$MODEL_VAL")
            ;;
        modalities_choice=*)
            MODALITIES_VAL="${arg#*=}"
            ARGS+=("data/modalities=$MODALITIES_VAL")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Additional arguments for W&B logging purposes
WANDB_LOG_ARGS=(
    "logger.wandb.group=RQ1_flaml_models"
    "tags=[RQ1,flaml,hours_slept,auc]"
)

# If swept parameters are resolved, construct a readable W&B run name
if [ -n "$MODEL_VAL" ] && [ -n "$SAMPLER_VAL" ] && [ -n "$MODALITIES_VAL" ]; then
    SHORT_SAMPLER="${SAMPLER_VAL#rolling_}"
    SHORT_MODEL="${MODEL_VAL#flaml_}"
    WANDB_LOG_ARGS+=("++logger.wandb.name=RQ1_${SHORT_MODEL}_${SHORT_SAMPLER}_${MODALITIES_VAL}")
fi

if [ "$HAS_UV_PREFIX" = true ]; then
    # Full 'uv run python src/train.py ...' was provided by the YAML command
    exec "${ARGS[@]}" "${WANDB_LOG_ARGS[@]}"
else
    # Fallback for direct manual execution (e.g. bash scripts/RQ1_flaml_models.sh sampler_pkg=... model_pkg=... modalities_choice=...)
    exec uv run python src/train.py \
        eval_plan=cyclical \
        task_name=cyclical_sweep \
        logger=wandb \
        data/scaler=dual \
        data.os_filter=both \
        data.collapse_strategy=none \
        data.use_demographics=false \
        data.use_sleep=true \
        data.sleep_feature_mode=hours_only \
        data.use_survey_context=false \
        data.require_sensor_data=true \
        data.purge_hours=120.0 \
        trainer.max_epochs=75 \
        trainer.enable_checkpointing=false \
        callbacks.model_checkpoint=null \
        callbacks.early_stopping=null \
        model.automl_config.metric=roc_auc \
        model.auto_class_weights=true \
        "${ARGS[@]}" \
        "${WANDB_LOG_ARGS[@]}"
fi
