#!/bin/bash
# Wrapper to map regression sweep parameters to Hydra config paths

ARGS=""

for arg in "$@"; do
    # Replace W&B parameter names with proper Hydra paths
    arg=${arg/aggregator_choice=/data/aggregator=}
    arg=${arg/scaler_choice=/data/scaler=}
    arg=${arg/sampler_choice=/data/sampler=}
    arg=${arg/resample_freq_choice=/++data.sampler.resample_freq=}
    
    # Wrap in quotes for safe shell passing
    ARGS="$ARGS \"$arg\""
done

# Execute the actual training script using eval to correctly expand quoted arguments
eval "uv run python src/train.py $ARGS"
