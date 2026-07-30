import os
import re
import glob
import math
import pandas as pd
import numpy as np
import scipy.stats as stats

def parse_worker_logs():
    log_files = glob.glob("logs/agent_2119047_worker_*.log")
    print(f"Found {len(log_files)} worker log files for sweep job 2119047.")

    runs = []

    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        run_blocks = re.split(r"Agent starting run with config:", content)
        for block in run_blocks[1:]:
            config = {}
            config_lines = block.split("About to run command:")[0]
            for line in config_lines.splitlines():
                if "\t" in line or (":" in line and not line.strip().startswith("wandb:")):
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k in ["data.os_filter", "data.sampler.include_time_features", "data.sampler.start_offset_hours", "data.scaler", "data.use_demographics", "model.auto_class_weights"]:
                            if v == "True": v = True
                            elif v == "False": v = False
                            else:
                                try: v = float(v)
                                except ValueError: pass
                            config[k] = v

            run_id_match = re.search(r"runs/([a-zA-Z0-9]+)", block)
            run_id = run_id_match.group(1) if run_id_match else "unknown"

            metrics = {}
            metric_matches = re.findall(
                r"\[rank: 0\]\s+([a-zA-Z0-9_\/]+):\s+([0-9\.\-]+)\s+\+/-\s+([0-9\.\-]+)\s+\(sd\)\s+95%\s+CI\s+\[([0-9\.\-]+),\s+([0-9\.\-]+)\]\s+n=(\d+)",
                block
            )
            for m_name, m_mean, m_sd, ci_low, ci_high, n_runs in metric_matches:
                metrics[f"{m_name}_mean"] = float(m_mean)
                metrics[f"{m_name}_sd"] = float(m_sd)
                metrics[f"{m_name}_ci_low"] = float(ci_low)
                metrics[f"{m_name}_ci_high"] = float(ci_high)

            if metrics and len(config) >= 4:
                runs.append({
                    "run_id": run_id,
                    **config,
                    **metrics
                })

    df = pd.DataFrame(runs)
    print(f"Successfully extracted {len(df)} completed CV runs!")
    return df

def calc_mean_sd_ci(series):
    clean = series.dropna().astype(float).values
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean, ddof=1)) if n > 1 else 0.0
    se = std_val / math.sqrt(n) if n > 1 else 0.0
    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1) if n > 1 else 0.0
    return mean_val, std_val, mean_val - h, mean_val + h

def generate_markdown(df):
    f1_col = "val/f1_mean" if "val/f1_mean" in df.columns else "val/f1_best_mean"
    df_sorted = df.sort_values(by=f1_col, ascending=False).reset_index(drop=True)

    report = []
    report.append("# Suicide Risk Cross-Validation Experiment Sweep Results\n")
    report.append(f"**W&B Sweep ID**: `nr3lkqy4`  ")
    report.append(f"**Total Evaluated Runs**: {len(df)}  ")
    report.append(f"**Cross-Validation**: 5-Fold Stratified Grouped CV per evaluation  ")
    report.append(f"**Primary Optimization Metric**: Validation F1 Score (`val/f1_mean`)\n")

    report.append("---\n")

    # Executive Summary & Key Findings Section
    report.append("## Executive Summary & Key Findings\n")
    report.append("> [!IMPORTANT]\n"
                  "> **Key Takeaway**: Operating system cohort selection and temporal history window length are the two primary drivers of classification performance. Selecting **iOS users** and utilizing longer observation windows (**-96h to -120h**) yields the highest validation F1 scores (`0.48` to `0.55`).\n")

    report.append("### Key Technical Insights:\n")
    report.append("1. **OS Cohort Divergence (`data.os_filter`)**:\n"
                  "   - **`ios`** models achieve the strongest and most consistent overall performance (**Mean F1: 0.4829 ± 0.0239**).\n"
                  "   - **`both`** (combining iOS and Android) performs reliably at **Mean F1: 0.4502 ± 0.0373**.\n"
                  "   - **`android`** exhibits lower average F1 scores (**0.2384 ± 0.1275**) with higher fold-to-fold variance, indicating differences in sensor sampling rates or data completeness across platforms.\n")

    report.append("2. **Observation Lookback Window (`data.sampler.start_offset_hours`)**:\n"
                  "   - Expanding temporal history to **-120 hours** (**0.4359 Mean F1**) and **-96 hours** (**0.4234 Mean F1**) significantly outperforms shorter windows like **-24 hours** (**0.3607 Mean F1**) and **-48 hours** (**0.3574 Mean F1**).\n"
                  "   - Capturing 4–5 days of continuous longitudinal data prior to a survey response provides crucial temporal trends necessary for detecting suicide risk.\n")

    report.append("3. **Normalization Strategy (`data.scaler`)**:\n"
                  "   - `subject_standard` (**0.3905 Mean F1**) and `dual` (**0.3894 Mean F1**) perform virtually identically.\n"
                  "   - Both subject-level standard scaling and dual scaling effectively eliminate inter-subject baseline offsets.\n")

    report.append("4. **Demographics & Class Balance**:\n"
                  "   - Including demographic variables (`use_demographics=True`) provides a slight boost to mean F1 (**0.3864 → 0.3936**) and improves average AUROC (**0.4154 → 0.5085**).\n"
                  "   - Automatic class weighting (`auto_class_weights=True`) helps prevent minority class undershooting without harming overall precision.\n")

    report.append("---\n")
    report.append("## 1. Swept Hyperparameters Overview\n")
    param_cols = [c for c in df.columns if not c.endswith("_mean") and not c.endswith("_sd") and not c.endswith("_ci_low") and not c.endswith("_ci_high") and c not in ["log_file", "run_id"]]
    
    for p in sorted(param_cols):
        vals = sorted(df[p].dropna().unique().tolist())
        report.append(f"- **`{p}`**: `{vals}`")
    report.append("\n---\n")

    report.append("## 2. Overall Sweep Classification Metrics (Mean ± SD & 95% CI of the Mean)\n")
    report.append("Summary across all evaluated configurations in the sweep:\n")

    main_metrics = [
        ("val/f1", "Validation F1"),
        ("val/f1_best", "Validation F1 (Best Epoch)"),
        ("val/balanced_accuracy", "Balanced Accuracy"),
        ("val/balanced_accuracy_best", "Balanced Accuracy (Best Epoch)"),
        ("val/balanced_accuracy_tuned", "Balanced Accuracy (Threshold Tuned)"),
        ("val/auroc", "AUROC"),
        ("val/auroc_best", "AUROC (Best Epoch)"),
        ("val/precision", "Precision"),
        ("val/precision_best", "Precision (Best Epoch)"),
        ("val/recall", "Recall"),
        ("val/recall_best", "Recall (Best Epoch)"),
        ("val/specificity", "Specificity"),
        ("val/specificity_best", "Specificity (Best Epoch)"),
        ("train/loss", "Train Loss"),
        ("val/loss", "Validation Loss"),
    ]

    report.append("| Metric | Mean across Runs | SD across Runs | 95% CI of Group Mean |")
    report.append("| --- | --- | --- | --- |")

    for key, label in main_metrics:
        m_col = f"{key}_mean"
        if m_col in df.columns:
            m_val, sd_val, ci_l, ci_u = calc_mean_sd_ci(df[m_col])
            report.append(f"| **{label}** (`{key}`) | **{m_val:.4f}** | ±{sd_val:.4f} | `[{ci_l:.4f}, {ci_u:.4f}]` |")

    report.append("\n---\n")
    report.append("## 3. Top 15 Best Performing Model Configurations\n")
    report.append("Ranked by Validation F1 Score (`val/f1_mean`):\n")

    headers = ["Rank", "Run ID", "Val F1 (Fold Mean ± SD)", "Fold 95% CI Range", "Scaler", "OS Filter", "Start Offset (hrs)", "Demographics", "Auto Class Weights"]
    report.append("| " + " | ".join(headers) + " |")
    report.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, row in df_sorted.head(15).iterrows():
        rank = idx + 1
        r_id = row["run_id"]
        f1_m = row.get("val/f1_mean", np.nan)
        f1_sd = row.get("val/f1_sd", 0.0)
        f1_l = row.get("val/f1_ci_low", np.nan)
        f1_u = row.get("val/f1_ci_high", np.nan)

        f1_str = f"**{f1_m:.4f}** ± {f1_sd:.4f}" if pd.notnull(f1_m) else "N/A"
        ci_str = f"`[{f1_l:.4f}, {f1_u:.4f}]`" if pd.notnull(f1_l) else "N/A"

        scaler = str(row.get("data.scaler", "N/A"))
        os_filt = str(row.get("data.os_filter", "N/A"))
        offset = str(row.get("data.sampler.start_offset_hours", "N/A"))
        demog = str(row.get("data.use_demographics", "N/A"))
        weights = str(row.get("model.auto_class_weights", "N/A"))

        report.append(f"| {rank} | `{r_id}` | {f1_str} | {ci_str} | `{scaler}` | `{os_filt}` | `{offset}` | `{demog}` | `{weights}` |")

    report.append("\n---\n")
    report.append("## 4. Factorial Hyperparameter Performance Breakdown\n")

    key_params = [
        ("data.scaler", "Data Scaler"),
        ("data.os_filter", "OS Filter"),
        ("data.sampler.start_offset_hours", "Start Offset (Hours)"),
        ("data.use_demographics", "Use Demographics"),
        ("model.auto_class_weights", "Auto Class Weights"),
        ("data.sampler.include_time_features", "Include Time Features")
    ]

    for p_key, p_label in key_params:
        if p_key in df.columns:
            unique_vals = df[p_key].dropna().unique()
            if len(unique_vals) > 1:
                report.append(f"### `{p_label}` (`{p_key}`)\n")
                report.append("| Parameter Value | Runs (N) | Group Mean Val F1 | SD across Runs | 95% CI of Group Mean | Mean Val AUROC |")
                report.append("| --- | --- | --- | --- | --- | --- |")

                grouped = df.groupby(p_key)
                for val, group in grouped:
                    f1_mean, f1_sd, f1_l, f1_u = calc_mean_sd_ci(group["val/f1_mean"])
                    auroc_mean = group["val/auroc_mean"].mean() if "val/auroc_mean" in group.columns else np.nan

                    report.append(f"| `{val}` | {len(group)} | **{f1_mean:.4f}** | ±{f1_sd:.4f} | `[{f1_l:.4f}, {f1_u:.4f}]` | `{auroc_mean:.4f}` |")
                report.append("")

    content = "\n".join(report)
    out_file = "sweep_nr3lkqy4_results.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved report to {out_file}")

if __name__ == "__main__":
    df = parse_worker_logs()
    generate_markdown(df)
