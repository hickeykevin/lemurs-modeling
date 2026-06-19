#!/bin/bash
# Wrapper to map W&B sweep parameters to Hydra config paths
# Usage: bash scripts/sweep_wrapper.sh aggregator_choice=suicide_risk ...

ARGS=""
IS_HEALTH_MODEL=false

# First pass: check which model we are using
for arg in "$@"; do
    if [[ "$arg" == "model_choice=health" ]]; then
        IS_HEALTH_MODEL=true
    fi
done

# Second pass: transform arguments
for arg in "$@"; do
    # Replace W&B parameter names with proper Hydra paths
    arg=${arg/model_choice=/model=}
    arg=${arg/aggregator_choice=/data/aggregator=}
    arg=${arg/sampler_choice=/data/sampler=}
    arg=${arg/scaler_choice=/data/scaler=}
    arg=${arg/modalities_choice=/data.modalities=}
    arg=${arg/split_mode_choice=/data.split_mode=}
    arg=${arg/resample_freq_choice=/++data.sampler.resample_freq=}
    
    # Only add LSTM-specific params if it's the 'health' model
    if [[ "$arg" == *"hidden_size_choice"* || "$arg" == *"num_layers_choice"* || "$arg" == *"lr_choice"* ]]; then
        if [ "$IS_HEALTH_MODEL" = true ]; then
            arg=${arg/hidden_size_choice=/model.net.hidden_size=}
            arg=${arg/num_layers_choice=/model.net.num_layers=}
            arg=${arg/lr_choice=/model.optimizer.lr=}
        else
            continue # Skip these for baseline models to avoid TypeErrors
        fi
    fi
    
    # Wrap in quotes for safe shell passing (crucial for lists like modalities)
    ARGS="$ARGS \"$arg\""
done

# Execute the actual training script
# Using eval to correctly interpret the quoted ARGS string
# We put $ARGS last so it can override the defaults
eval "uv run python src/train.py \
    logger=wandb \
    trainer=cpu \
    trainer.max_epochs=50 \
    data.os_filter=ios \
    'callbacks=[classification_metrics,confusion_matrix]' \
    callbacks.confusion_matrix.frequency=5 \
    data.modalities=["step"] \
    $ARGS"
