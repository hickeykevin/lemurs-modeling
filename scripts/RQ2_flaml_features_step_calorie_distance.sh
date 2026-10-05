#!/bin/bash
# Wrapper script for RQ2 FLAML Features sweep (all 3 modalities: step, calorie, distance)
# Translates W&B sweep parameters into Hydra config overrides and attaches W&B logging metadata

set -eo pipefail

ARGS=()
SAMPLER_VAL=""
MODEL_VAL=""
FEATURIZER_VAL=""
HAS_UV_PREFIX=false

# Check if command line starts with 'uv' (as configured in RQ2_flaml_features_step_calorie_distance.yaml)
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
        featurizer_pkg=*)
            FEATURIZER_VAL="${arg#*=}"
            ARGS+=("model/featurizer=$FEATURIZER_VAL")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Additional arguments for W&B logging purposes
WANDB_LOG_ARGS=(
    "logger.wandb.group=RQ2_flaml_features_step_calorie_distance"
    "tags=[RQ2,flaml,features,step_calorie_distance,auc]"
)

# If swept parameters are resolved, construct a readable W&B run name
if [ -n "$MODEL_VAL" ] && [ -n "$SAMPLER_VAL" ] && [ -n "$FEATURIZER_VAL" ]; then
    SHORT_SAMPLER="${SAMPLER_VAL#rolling_}"
    SHORT_MODEL="${MODEL_VAL#flaml_}"
    WANDB_LOG_ARGS+=("++logger.wandb.name=RQ2_${SHORT_MODEL}_${SHORT_SAMPLER}_step_calorie_distance_${FEATURIZER_VAL}")
fi

if [ "$HAS_UV_PREFIX" = true ]; then
    # Full 'uv run python src/train.py ...' was provided by the YAML command
    exec "${ARGS[@]}" "${WANDB_LOG_ARGS[@]}"
else
    # Fallback for direct manual execution (e.g. bash scripts/RQ2_flaml_features_step_calorie_distance.sh sampler_pkg=... model_pkg=... featurizer_pkg=...)
    exec uv run python src/train.py \
        eval_plan=cyclical \
        task_name=cyclical_sweep \
        logger=wandb \
        data/modalities=step_calorie_distance \
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
