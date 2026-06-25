import os
import pandas as pd
import numpy as np
import rootutils
import wandb

# Initialize rootutils
root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def main():
    api = wandb.Api()
    
    # 1. Determine entity
    try:
        entity = api.viewer.entity
        print(f"Logged in user/entity: {entity}")
    except Exception as e:
        print(f"Could not get viewer entity automatically: {e}")
        entity = None
        
    project_name = "lemurs-modeling"
    project_path = f"{entity}/{project_name}" if entity else project_name
    print(f"Querying project: {project_path}")
    
    import sys
    cli_sweep_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    target_sweep = None
    target_sweep_name = "regression_hyperparameter_sweep"
    
    if cli_sweep_id:
        print(f"Using sweep ID from command line: {cli_sweep_id}")
        try:
            path = f"{entity}/{project_name}/{cli_sweep_id}" if entity else f"{project_name}/{cli_sweep_id}"
            target_sweep = api.sweep(path)
        except Exception as e:
            print(f"Error fetching sweep {cli_sweep_id}: {e}")
    else:
        try:
            project = api.project(project_path)
            sweeps = project.sweeps()
        except Exception as e:
            print(f"Error fetching sweeps for project {project_path}: {e}")
            sweeps = []
            
        print("\nAvailable sweeps:")
        matching_sweeps = []
        for sw in sweeps:
            name = sw.config.get("name", "") if isinstance(sw.config, dict) else getattr(sw, "name", "")
            print(f"  ID: {sw.id} | Name: {name} | State: {sw.state}")
            if name == target_sweep_name:
                matching_sweeps.append(sw)
                
        if matching_sweeps:
            target_sweep = matching_sweeps[0]
            print(f"\nSelecting sweep '{target_sweep_name}' (ID: {target_sweep.id})")
            if len(matching_sweeps) > 1:
                print(f"Note: Found {len(matching_sweeps)} sweeps with this name. To analyze a different one, pass its ID as an argument.")
            
        # Try using local sweep IDs if we couldn't fetch project sweeps
        if not target_sweep:
            print("\nChecking local sweep IDs as fallback...")
            local_sweep_ids = ["a80vjd82", "gwpadm3n", "p3l5o9se", "p76flg5h", "qnr4ihx9", "tggt1e0f"]
            for sid in local_sweep_ids:
                try:
                    path = f"{entity}/{project_name}/{sid}" if entity else f"{project_name}/{sid}"
                    sw = api.sweep(path)
                    name = sw.config.get("name", "") if isinstance(sw.config, dict) else getattr(sw, "name", "")
                    print(f"  Successfully retrieved local sweep ID: {sid} | Name: {name} | State: {sw.state}")
                    if name == target_sweep_name:
                        target_sweep = sw
                        break
                except Exception as ex:
                    pass

        if not target_sweep and sweeps:
            print(f"\nCould not find sweep named '{target_sweep_name}' exactly.")
            # Search for any sweep with 'regression' in the name
            for sw in sweeps:
                name = sw.config.get("name", "") if isinstance(sw.config, dict) else getattr(sw, "name", "")
                if "regression" in name.lower():
                    print(f"Selecting sweep '{name}' (ID: {sw.id}) as candidate.")
                    target_sweep = sw
                    break
            if not target_sweep:
                target_sweep = sweeps[0]
                print(f"Defaulting to first sweep in list: '{target_sweep.config.get('name')}' (ID: {target_sweep.id})")
            
    if not target_sweep:
        print("\nNo sweeps found. Please make sure you are logged in to wandb (run 'wandb login') and specify the sweep ID manually.")
        return
        
    print(f"\nAnalyzing sweep: {target_sweep.config.get('name', 'N/A')} (ID: {target_sweep.id})")
    
    # 2. Retrieve all runs
    runs = target_sweep.runs
    print(f"Total runs in sweep: {len(runs)}")
    
    # 3. Extract parameters and metrics
    runs_data = []
    for run in runs:
        config = run.config
        summary = run.summary._json_dict
        
        run_dict = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
        }
        
        for k, v in config.items():
            run_dict[f"config/{k}"] = v
        for k, v in summary.items():
            run_dict[f"metric/{k}"] = v
            
        runs_data.append(run_dict)
        
    df = pd.DataFrame(runs_data)
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    csv_path = os.path.join(root_dir, "notebooks", f"sweep_{target_sweep.id}_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep data to {csv_path}")
    
    # Let's perform basic analysis on 'suicide_risk_regression'
    # Find the correct column names for aggregator and metric
    agg_cols = [c for c in df.columns if "aggregator" in c]
    mse_cols = [c for c in df.columns if "val/mse" in c or (c.startswith("metric/") and "mse" in c)]
    
    print("\n--- Detected Columns ---")
    print(f"Aggregator columns: {agg_cols}")
    print(f"MSE columns: {mse_cols}")
    
    if not agg_cols:
        print("Could not find aggregator column.")
        return
    if not mse_cols:
        print("Could not find MSE metric column.")
        return
        
    agg_col = agg_cols[0]
    mse_col = mse_cols[0]
    
    # Filter for suicide_risk_regression
    df_suicide = df[df[agg_col] == "suicide_risk_regression"]
    print(f"\nNumber of suicide_risk_regression runs: {len(df_suicide)}")
    
    if len(df_suicide) == 0:
        print("No runs found with aggregator = suicide_risk_regression.")
        # Show unique aggregators available
        print(f"Available aggregators: {df[agg_col].unique() if agg_col in df.columns else 'None'}")
        return
        
    # Clean and convert MSE to numeric
    df_suicide[mse_col] = pd.to_numeric(df_suicide[mse_col], errors='coerce')
    df_suicide_valid = df_suicide.dropna(subset=[mse_col])
    print(f"Runs with valid MSE metrics: {len(df_suicide_valid)}")
    
    if len(df_suicide_valid) == 0:
        print("No suicide_risk_regression runs have a valid val/mse metric.")
        return
        
    # Top runs
    df_suicide_sorted = df_suicide_valid.sort_values(by=mse_col)
    config_cols = [c for c in df.columns if c.startswith("config/") and c != agg_col]
    
    # Filter config_cols to only keep columns that are hashable and have multiple values
    valid_config_cols = []
    for col in config_cols:
        has_unhashable = df_suicide_valid[col].apply(lambda x: isinstance(x, (dict, list, set))).any()
        if not has_unhashable:
            try:
                unique_vals = df_suicide_valid[col].dropna().unique()
                if len(unique_vals) > 1:
                    valid_config_cols.append(col)
            except Exception:
                pass

    print(f"\n=== Top 10 suicide_risk_regression runs (by {mse_col}) ===")
    print(df_suicide_sorted[["run_name", mse_col] + valid_config_cols].head(10).to_string(index=False))
    
    # Main effects
    print(f"\n=== Main Effects on {mse_col} for suicide_risk_regression ===")
    for col in valid_config_cols:
        grouped = df_suicide_valid.groupby(col)[mse_col].agg(["count", "mean", "min", "std"]).reset_index()
        # Sort by mean MSE ascending
        grouped = grouped.sort_values(by="mean")
        print(f"\n--- Hyperparameter: {col.replace('config/', '')} ---")
        print(grouped.to_string(index=False))

if __name__ == "__main__":
    main()
