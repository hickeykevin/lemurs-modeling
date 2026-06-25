import os
import pandas as pd
import numpy as np
import rootutils

# Initialize rootutils
root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def main():
    csv_path = os.path.join(root_dir, "notebooks", "sweep_u929626r_data.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Column detection
    agg_col = "config/data/aggregator"
    mse_col = "metric/val/mse"
    
    # Check if columns exist
    if agg_col not in df.columns or mse_col not in df.columns:
        print("Missing required columns in CSV.")
        print(f"Columns in CSV: {df.columns.tolist()}")
        return

    # Filter for suicide_risk_regression
    df_suicide = df[df[agg_col] == "suicide_risk_regression"].copy()
    df_suicide[mse_col] = pd.to_numeric(df_suicide[mse_col], errors='coerce')
    df_suicide = df_suicide.dropna(subset=[mse_col])
    
    print(f"Loaded {len(df_suicide)} runs for suicide_risk_regression.")
    
    # Define hyperparameter columns
    scaler_col = "config/data/scaler"
    sampler_col = "config/data/sampler"
    os_col = "config/data.os_filter"
    collapse_col = "config/data.collapse_strategy"
    lookback_col = "config/++data.sampler.lookback_days"
    freq_col = "config/++data.sampler.resample_freq"
    offset_col = "config/++data.sampler.start_offset_hours"
    
    print("\n" + "="*80)
    print(" SECTION 1: iOS ONLY ANALYSIS")
    print("="*80)
    print("Since 'both' OS filter performs very poorly (mean MSE ~1.06 vs iOS ~0.63),")
    print("we isolate runs with config/data.os_filter == 'ios' to remove skew.")
    
    df_ios = df_suicide[df_suicide[os_col] == "ios"].copy()
    print(f"Number of iOS-only suicide_risk_regression runs: {len(df_ios)}")
    
    # Main effects within iOS-only runs
    features = [
        ("Scaler", scaler_col),
        ("Sampler", sampler_col),
        ("Collapse Strategy", collapse_col),
        ("Lookback Days", lookback_col),
        ("Resample Frequency", freq_col),
        ("Start Offset Hours", offset_col)
    ]
    
    for label, col in features:
        if col in df_ios.columns:
            grouped = df_ios.groupby(col)[mse_col].agg(["count", "mean", "min", "std"]).reset_index()
            grouped = grouped.sort_values(by="mean")
            print(f"\n--- {label} ({col.replace('config/', '')}) ---")
            print(grouped.to_string(index=False))
            
    print("\n" + "="*80)
    print(" SECTION 2: iOS + MEAN COLLAPSE STRATEGY ANALYSIS")
    print("="*80)
    print("Further isolating to the best collapse strategy (config/data.collapse_strategy == 'mean').")
    
    df_ios_mean = df_ios[df_ios[collapse_col] == "mean"].copy()
    print(f"Number of runs with iOS + Mean Collapse: {len(df_ios_mean)}")
    
    features_sub = [
        ("Scaler", scaler_col),
        ("Sampler", sampler_col),
        ("Lookback Days", lookback_col),
        ("Resample Frequency", freq_col),
        ("Start Offset Hours", offset_col)
    ]
    
    for label, col in features_sub:
        if col in df_ios_mean.columns:
            grouped = df_ios_mean.groupby(col)[mse_col].agg(["count", "mean", "min", "std"]).reset_index()
            grouped = grouped.sort_values(by="mean")
            print(f"\n--- {label} ({col.replace('config/', '')}) ---")
            print(grouped.to_string(index=False))
            
    print("\n" + "="*80)
    print(" SECTION 3: INTERACTION EFFECTS (iOS + Mean Collapse runs)")
    print("="*80)
    
    # Interaction: Scaler x Sampler
    print("\n--- Interaction: Scaler x Sampler ---")
    if scaler_col in df_ios_mean.columns and sampler_col in df_ios_mean.columns:
        grouped = df_ios_mean.groupby([scaler_col, sampler_col])[mse_col].agg(["count", "mean", "min"]).reset_index()
        grouped = grouped.sort_values(by="mean")
        print(grouped.to_string(index=False))
        
    # Interaction: Sampler x Lookback Days
    print("\n--- Interaction: Sampler x Lookback Days ---")
    if sampler_col in df_ios_mean.columns and lookback_col in df_ios_mean.columns:
        grouped = df_ios_mean.groupby([sampler_col, lookback_col])[mse_col].agg(["count", "mean", "min"]).reset_index()
        grouped = grouped.sort_values(by="mean")
        print(grouped.to_string(index=False))

    # Interaction: Resample Frequency x Start Offset Hours
    print("\n--- Interaction: Resample Frequency x Start Offset Hours ---")
    if freq_col in df_ios_mean.columns and offset_col in df_ios_mean.columns:
        grouped = df_ios_mean.groupby([freq_col, offset_col])[mse_col].agg(["count", "mean", "min"]).reset_index()
        grouped = grouped.sort_values(by="mean")
        print(grouped.to_string(index=False))

    # Best overall combinations
    print("\n" + "="*80)
    print(" SECTION 4: TOP 15 RUNS (iOS + Mean Collapse)")
    print("="*80)
    df_ios_mean_sorted = df_ios_mean.sort_values(by=mse_col)
    cols_to_print = ["run_name", mse_col, scaler_col, sampler_col, lookback_col, freq_col, offset_col]
    # Clean column names for printing
    df_print = df_ios_mean_sorted[cols_to_print].head(15).copy()
    df_print.columns = [c.split("/")[-1].split(".")[-1] for c in cols_to_print]
    print(df_print.to_string(index=False))

if __name__ == "__main__":
    main()
