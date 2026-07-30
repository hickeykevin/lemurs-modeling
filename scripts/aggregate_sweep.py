import sys
import wandb
import pandas as pd
import numpy as np
import scipy.stats as stats

def get_ci95(data):
    clean = [float(x) for x in data if pd.notnull(x) and not np.isnan(float(x))]
    if len(clean) < 2:
        if len(clean) == 1:
            return clean[0], 0.0, clean[0], clean[0]
        return 0.0, 0.0, 0.0, 0.0
    n = len(clean)
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1) if n > 1 else 0.0
    return mean, h, mean - h, mean + h

api = wandb.Api()
sweep = api.sweep("hickeykevin/lemurs-modeling/sweeps/nr3lkqy4")
print("Sweep:", sweep.name, "State:", sweep.state)

records = []
count = 0
for r in sweep.runs:
    count += 1
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in r.config.items() if not k.startswith("_")}
    summary = {k: float(v) for k, v in r.summary.items() if isinstance(v, (int, float, np.number)) and not k.startswith("_")}
    records.append({"run_id": r.id, "run_name": r.name, "state": r.state, **cfg, **summary})
    if count % 20 == 0:
        print(f"Loaded {count} runs...", flush=True)

df = pd.DataFrame(records)
print(f"Total processed runs: {len(df)}")
df.to_csv("logs/sweep_nr3lkqy4_parsed.csv", index=False)

f1_col = "val/f1" if "val/f1" in df.columns else [c for c in df.columns if "f1" in c][0]
df_sorted = df.sort_values(by=f1_col, ascending=False).reset_index(drop=True)

report = []
report.append(f"# Suicide Risk CV Experiment Sweep Summary (`nr3lkqy4`)\n")
report.append(f"- **W&B Sweep ID**: `nr3lkqy4`")
report.append(f"- **Sweep Name**: `{sweep.name}`")
report.append(f"- **Total Runs**: {len(df)}")
report.append(f"- **Primary Metric**: `{f1_col}` (maximize)\n")

report.append("---\n")
report.append("## 1. Swept Hyperparameters\n")
param_cols = [c for c in df.columns if c not in ["run_id", "run_name", "state"] and not c.startswith("val/") and not c.startswith("cv_summary/") and not c.startswith("cv/")]
for p in sorted(param_cols):
    vals = df[p].dropna().unique().tolist()
    report.append(f"- **`{p}`**: `{vals}`")

report.append("\n---\n")
report.append("## 2. Overall Summary & Confidence Intervals\n")
f1_m, f1_h, f1_l, f1_u = get_ci95(df[f1_col])
report.append(f"- **Validation F1 (`{f1_col}`)**: **{f1_m:.4f} ± {f1_h:.4f}** (95% CI: `[{f1_l:.4f}, {f1_u:.4f}]`)")

other_metrics = [c for c in df.columns if c.startswith("val/") and c != f1_col]
for m in sorted(other_metrics):
    mm, mh, ml, mu = get_ci95(df[m])
    report.append(f"- **`{m}`**: **{mm:.4f} ± {mh:.4f}** (95% CI: `[{ml:.4f}, {mu:.4f}]`)")

report.append("\n---\n")
report.append("## 3. Top 10 Highest F1 Score Runs\n")
headers = ["Rank", "Run ID", "Val F1", "Scaler", "OS Filter", "Start Offset (hrs)", "Demographics", "Auto Class Weights"]
report.append("| " + " | ".join(headers) + " |")
report.append("| " + " | ".join(["---"] * len(headers)) + " |")

for idx, row in df_sorted.head(10).iterrows():
    val_f1 = f"{row[f1_col]:.4f}" if pd.notnull(row[f1_col]) else "N/A"
    scaler = str(row.get("data.scaler", row.get("data/scaler", "N/A")))
    os_filt = str(row.get("data.os_filter", "N/A"))
    offset = str(row.get("data.sampler.start_offset_hours", "N/A"))
    demog = str(row.get("data.use_demographics", "N/A"))
    weights = str(row.get("model.auto_class_weights", "N/A"))
    report.append(f"| {idx+1} | `{row['run_id']}` | **{val_f1}** | `{scaler}` | `{os_filt}` | `{offset}` | `{demog}` | `{weights}` |")

report.append("\n---\n")
report.append("## 4. Parameter Performance Breakdown (Mean ± 95% CI)\n")

for p in param_cols:
    if len(df[p].dropna().unique()) > 1:
        report.append(f"### Influence of `{p}`\n")
        report.append("| Value | N Runs | Mean Val F1 | 95% CI Margin | 95% CI Range |")
        report.append("| --- | --- | --- | --- | --- |")
        for val, scores in df.groupby(p)[f1_col].apply(list).items():
            m, h, l, u = get_ci95(scores)
            report.append(f"| `{val}` | {len(scores)} | **{m:.4f}** | ±{h:.4f} | [{l:.4f}, {u:.4f}] |")
        report.append("")

with open("sweep_nr3lkqy4_results.md", "w") as f:
    f.write("\n".join(report))

print("Saved sweep_nr3lkqy4_results.md successfully!")
