import json
import math
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import wandb

def get_ci95(data):
    clean = [float(x) for x in data if pd.notnull(x) and not math.isnan(float(x))]
    if len(clean) < 2:
        if len(clean) == 1:
            return clean[0], 0.0, clean[0], clean[0]
        return 0.0, 0.0, 0.0, 0.0
    n = len(clean)
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    se = std / math.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1) if n > 1 else 0.0
    return mean, h, mean - h, mean + h

api = wandb.Api()
api_key = api.api_key
headers = {"Authorization": f"Bearer {api_key}"}

query = """
query SweepRuns($entity: String!, $project: String!, $sweep: String!) {
  project(name: $project, entityName: $entity) {
    sweep(sweepName: $sweep) {
      id
      name
      runs {
        edges {
          node {
            id
            name
            state
            config
            summaryMetrics
          }
        }
      }
    }
  }
}
"""

variables = {
    "entity": "hickeykevin",
    "project": "lemurs-modeling",
    "sweep": "nr3lkqy4"
}

print("Sending direct GraphQL request to W&B API...")
response = requests.post("https://api.wandb.ai/graphql", json={"query": query, "variables": variables}, headers=headers)
data = response.json()

sweep_data = data["data"]["project"]["sweep"]
print(f"Sweep ID: {sweep_data['id']}, Name: {sweep_data['name']}")

edges = sweep_data["runs"]["edges"]
print(f"Retrieved {len(edges)} runs via GraphQL!")

records = []
for edge in edges:
    node = edge["node"]
    cfg_raw = json.loads(node["config"]) if isinstance(node["config"], str) else node["config"]
    sum_raw = json.loads(node["summaryMetrics"]) if isinstance(node["summaryMetrics"], str) else node["summaryMetrics"]

    cfg = {}
    for k, v in cfg_raw.items():
        if k.startswith("_"):
            continue
        cfg[k] = v["value"] if isinstance(v, dict) and "value" in v else v

    summary = {}
    for k, v in sum_raw.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, np.number)):
            summary[k] = float(v)

    records.append({
        "run_id": node["id"],
        "run_name": node["name"],
        "state": node["state"],
        **cfg,
        **summary
    })

df = pd.DataFrame(records)
print(f"DataFrame created with shape: {df.shape}")
df.to_csv("logs/sweep_nr3lkqy4_parsed.csv", index=False)

f1_col = "val/f1" if "val/f1" in df.columns else ([c for c in df.columns if "f1" in c] + ["val/f1"])[0]
df_sorted = df.sort_values(by=f1_col, ascending=False).reset_index(drop=True)

param_cols = [c for c in df.columns if c not in ["run_id", "run_name", "state"] and not c.startswith("val/") and not c.startswith("cv_summary/") and not c.startswith("cv/")]

report = []
report.append(f"# Suicide Risk Cross-Validation Experiment Sweep Summary (`nr3lkqy4`)\n")
report.append(f"- **W&B Sweep ID**: `nr3lkqy4`")
report.append(f"- **Sweep Name**: `{sweep_data['name']}`")
report.append(f"- **Total Runs Analyzed**: {len(df)}")
report.append(f"- **Optimization Metric**: `{f1_col}` (maximize)\n")

report.append("---\n")
report.append("## 1. Swept Hyperparameters Overview\n")
for p in sorted(param_cols):
    unique_vals = df[p].dropna().unique().tolist()
    report.append(f"- **`{p}`**: `{unique_vals}`")

report.append("\n---\n")
report.append("## 2. Overall Summary Statistics & Confidence Intervals\n")

f1_m, f1_h, f1_l, f1_u = get_ci95(df[f1_col])
report.append(f"Across all {len(df)} sweep runs:\n")
report.append(f"- **Validation F1 (`{f1_col}`)**: **{f1_m:.4f} ± {f1_h:.4f}** (95% CI: `[{f1_l:.4f}, {f1_u:.4f}]`)")

metric_cols = sorted([c for c in df.columns if c.startswith("val/") and c != f1_col])
for m in metric_cols:
    mm, mh, ml, mu = get_ci95(df[m])
    report.append(f"- **`{m}`**: **{mm:.4f} ± {mh:.4f}** (95% CI: `[{ml:.4f}, {mu:.4f}]`)")

report.append("\n---\n")
report.append("## 3. Leaderboard: Top Highest F1 Score Runs\n")

headers = ["Rank", "Run ID", "Val F1", "Scaler", "OS Filter", "Start Offset (hrs)", "Demographics", "Auto Class Weights"]
report.append("| " + " | ".join(headers) + " |")
report.append("| " + " | ".join(["---"] * len(headers)) + " |")

top_10 = df_sorted.head(15)
for idx, row in top_10.iterrows():
    val_f1 = f"{row[f1_col]:.4f}" if pd.notnull(row[f1_col]) else "N/A"
    scaler = str(row.get("data.scaler", row.get("data/scaler", "N/A")))
    os_filt = str(row.get("data.os_filter", "N/A"))
    offset = str(row.get("data.sampler.start_offset_hours", "N/A"))
    demog = str(row.get("data.use_demographics", "N/A"))
    weights = str(row.get("model.auto_class_weights", "N/A"))
    report.append(f"| {idx+1} | `{row['run_id']}` | **{val_f1}** | `{scaler}` | `{os_filt}` | `{offset}` | `{demog}` | `{weights}` |")

report.append("\n---\n")
report.append("## 4. Factorial Parameter Breakdown (Mean ± 95% CI)\n")

for p in sorted(param_cols):
    vals = df[p].dropna().unique()
    if len(vals) > 1:
        report.append(f"### Metric Breakdown by `{p}`\n")
        report.append("| Parameter Value | Runs (N) | Mean Val F1 | 95% CI Margin | 95% CI Range |")
        report.append("| --- | --- | --- | --- | --- |")
        grouped = df.groupby(p)[f1_col].apply(list)
        for val, scores in grouped.items():
            m, h, l, u = get_ci95(scores)
            report.append(f"| `{val}` | {len(scores)} | **{m:.4f}** | ±{h:.4f} | [{l:.4f}, {u:.4f}] |")
        report.append("")

output_path = "sweep_nr3lkqy4_results.md"
with open(output_path, "w") as f:
    f.write("\n".join(report))

print(f"Generated {output_path} successfully!")
