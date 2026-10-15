#!/bin/bash
# Wrapper for the foundation-model W&B sweeps (TabPFN-3.5 frozen / fine-tuned, TiRex frozen /
# fine-tuned, TiRex as a featurizer). Same role as RQ1_flaml_models.sh / RQ1_DL_models.sh:
# translates sweep parameters into Hydra overrides and sets the W&B group, tags and run name.
# Sweep parameters: sampler_pkg, model_pkg, modalities_choice, featurizer_pkg (optional),
# wandb_group (consumed here, not passed to Hydra).
set -eo pipefail

ARGS=()
SAMPLER_VAL=""; MODEL_VAL=""; MODALITIES_VAL=""; FEATURIZER_VAL=""; GROUP="RQ_foundation_models"
for arg in "$@"; do
    case "$arg" in
        sampler_pkg=*)       SAMPLER_VAL="${arg#*=}";    ARGS+=("data/sampler=sweep_configs/$SAMPLER_VAL") ;;
        model_pkg=*)         MODEL_VAL="${arg#*=}";      ARGS+=("model=$MODEL_VAL") ;;
        modalities_choice=*) MODALITIES_VAL="${arg#*=}"; ARGS+=("data/modalities=$MODALITIES_VAL") ;;
        featurizer_pkg=*)    FEATURIZER_VAL="${arg#*=}"; ARGS+=("model/featurizer=$FEATURIZER_VAL") ;;
        wandb_group=*)       GROUP="${arg#*=}" ;;
        *)                   ARGS+=("$arg") ;;
    esac
done

WANDB_LOG_ARGS=("logger.wandb.group=$GROUP" "tags=[${GROUP},foundation,hours_slept,auc]")
if [ -n "$MODEL_VAL" ] && [ -n "$SAMPLER_VAL" ]; then
    NAME="${GROUP}_${MODEL_VAL#flaml_}_${SAMPLER_VAL#rolling_}"
    [ -n "$MODALITIES_VAL" ] && NAME="${NAME}_${MODALITIES_VAL}"
    [ -n "$FEATURIZER_VAL" ] && NAME="${NAME}_${FEATURIZER_VAL}"
    WANDB_LOG_ARGS+=("++logger.wandb.name=$NAME")
fi
# The YAML command provides the full 'uv run python src/train.py <fixed settings>' prefix.
exec "${ARGS[@]}" "${WANDB_LOG_ARGS[@]}"
