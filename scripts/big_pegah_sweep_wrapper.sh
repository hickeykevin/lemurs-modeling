#!/bin/bash
# Wrapper script for Big Pegah Sweep to translate W&B parameters to Hydra overrides safely.
# Dynamically filters model-specific parameters to avoid Hydra composition errors (e.g. model/net on flaml).

ARGS=""
IS_DEFAULT_MODEL=true

# First pass: check model selection
for arg in "$@"; do
    if [[ "$arg" == "model_choice=flaml" ]]; then
        IS_DEFAULT_MODEL=false
    fi
done

# Second pass: map choices to Hydra override syntax
for arg in "$@"; do
    case "$arg" in
        model_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"model=$val\""
            ;;
        net_choice=*)
            if [ "$IS_DEFAULT_MODEL" = true ]; then
                val="${arg#*=}"
                ARGS="$ARGS \"model/net=$val\""
            fi
            ;;
        pooling_choice=*)
            if [ "$IS_DEFAULT_MODEL" = true ]; then
                val="${arg#*=}"
                ARGS="$ARGS \"++model.net.pooling=$val\""
            fi
            ;;
        scaler_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data/scaler=$val\""
            ;;
        sampler_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data/sampler=$val\""
            ;;
        start_offset_hours_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++data.sampler.start_offset_hours=$val\""
            ;;
        lookback_hours_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++data.sampler.lookback_hours=$val\""
            ;;
        resample_freq_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++data.sampler.resample_freq=$val\""
            ;;
        bin_edges_hours_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++data.sampler.bin_edges_hours=$val\""
            ;;
        modalities_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.modalities=$val\" \"data/preprocessors=$val\""
            ;;
        os_filter_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.os_filter=$val\""
            ;;
        collapse_strategy_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.collapse_strategy=$val\""
            ;;
        require_sensor_data_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.require_sensor_data=$val\""
            ;;
        use_survey_context_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.use_survey_context=$val\""
            ;;
        include_time_features_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.include_time_features=$val\" \"++data.sampler.include_time_features=$val\""
            ;;
        use_demographics_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.use_demographics=$val\""
            ;;
        use_sleep_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"data.use_sleep=$val\""
            ;;
        weight_decay_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++model.optimizer.weight_decay=$val\""
            ;;
        class_weights_choice=*)
            val="${arg#*=}"
            ARGS="$ARGS \"++model.class_weights=$val\""
            ;;
        *)
            ARGS="$ARGS \"$arg\""
            ;;
    esac
done

# Execute training script with base configs and sanitized overrides
eval "uv run python src/train.py \
    task_name=big_pegah_sweep \
    paths.log_dir=\${paths.root_dir}/logs/big_pegah_sweep \
    data/aggregator=suicide_risk \
    data.split_mode=longitudinal \
    seed=7 \
    trainer=cpu \
    trainer.max_epochs=100 \
    'callbacks=[classification_metrics,confusion_matrix,model_checkpoint]' \
    callbacks.model_checkpoint.monitor=val/f1 \
    callbacks.model_checkpoint.mode=max \
    callbacks.model_checkpoint.save_top_k=1 \
    callbacks.model_checkpoint.save_last=false \
    callbacks.confusion_matrix.frequency=5 \
    'logger=[csv,wandb]' \
    logger.wandb.group=big_pegah_sweep \
    model.auto_class_weights=false \
    $ARGS"

