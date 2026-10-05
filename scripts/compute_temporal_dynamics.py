"""
Compute Temporal Dynamics (Marginal Lookback, Marginal Window, and 2D Heatmaps)
for Feature Ablation runs matching the structure of flaml_sweep_report.md L172-L221.
Runs purely with standard library for instant execution.
"""

import csv
import json
import math
import re
from pathlib import Path

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def analyze_subset(records, name=""):
    lookbacks = ['24h', '48h', '72h', '96h', '120h']
    windows = ['4h', '6h', '8h', '12h']

    # 1. Marginal Lookback
    lb_stats = {}
    for lb in lookbacks:
        matching = [r for r in records if r['lookback'] == lb]
        aurocs = [r['auroc'] for r in matching]
        f1s = [r['f1'] for r in matching]
        baccs = [r['bacc'] for r in matching]
        lb_stats[lb] = {
            'count': len(matching),
            'f1_mean': mean(f1s),
            'f1_std': std(f1s),
            'f1_med': median(f1s),
            'auroc_mean': mean(aurocs),
            'auroc_std': std(aurocs),
            'auroc_med': median(aurocs),
            'bacc_mean': mean(baccs),
            'bacc_std': std(baccs),
        }

    # 2. Marginal Window
    win_stats = {}
    for win in windows:
        matching = [r for r in records if r['window'] == win]
        aurocs = [r['auroc'] for r in matching]
        f1s = [r['f1'] for r in matching]
        baccs = [r['bacc'] for r in matching]
        win_stats[win] = {
            'count': len(matching),
            'f1_mean': mean(f1s),
            'f1_std': std(f1s),
            'f1_med': median(f1s),
            'auroc_mean': mean(aurocs),
            'auroc_std': std(aurocs),
            'auroc_med': median(aurocs),
            'bacc_mean': mean(baccs),
            'bacc_std': std(baccs),
        }

    # 3. 2D Heatmap: AUROC & F1
    heatmap_auroc = {lb: {} for lb in lookbacks}
    heatmap_f1 = {lb: {} for lb in lookbacks}
    for lb in lookbacks:
        for win in windows:
            matching = [r for r in records if r['lookback'] == lb and r['window'] == win]
            aurocs = [r['auroc'] for r in matching]
            f1s = [r['f1'] for r in matching]
            heatmap_auroc[lb][win] = mean(aurocs) if aurocs else None
            heatmap_f1[lb][win] = mean(f1s) if f1s else None

    return {
        'name': name,
        'count': len(records),
        'lookback_marginal': lb_stats,
        'window_marginal': win_stats,
        'heatmap_auroc': heatmap_auroc,
        'heatmap_f1': heatmap_f1,
    }

def main():
    csv_file = Path('reports/feature_ablation/summary.csv')
    records = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'SUCCESS':
                continue
            sampler = row['sampler'].replace('.yaml', '')
            m = re.match(r'rolling_(\d+)h_(\d+)h', sampler)
            if not m:
                continue
            lb = f"{m.group(1)}h"
            win = f"{m.group(2)}h"
            records.append({
                'condition_id': int(row['id']),
                'condition_name': row['name'],
                'sampler': sampler,
                'lookback': lb,
                'window': win,
                'modality': row['modality'].replace('.yaml', ''),
                'auroc': float(row['auroc']),
                'f1': float(row['f1']),
                'bacc': float(row['balanced_acc']) if row['balanced_acc'] else 0.0,
            })

    print(f"Loaded {len(records)} successful records for temporal dynamics analysis.")

    # 1. Overall (All Conditions)
    res_overall = analyze_subset(records, "Overall (All Conditions)")
    
    # 2. Condition 0 (Pure Sensor Baseline)
    res_cond0 = analyze_subset([r for r in records if r['condition_id'] == 0], "Condition 0: Pure Sensor Baseline")

    # 3. Condition 4 (All Demographics)
    res_cond4 = analyze_subset([r for r in records if r['condition_id'] == 4], "Condition 4: All Demographics")

    output_data = {
        'overall': res_overall,
        'condition_0': res_cond0,
        'condition_4': res_cond4,
    }

    out_json = Path('reports/temporal_dynamics.json')
    out_json.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
    print(f"Saved temporal dynamics to {out_json}")

    # Generate a markdown snippet
    lookbacks = ['24h', '48h', '72h', '96h', '120h']
    windows = ['4h', '6h', '8h', '12h']

    def format_tables(res, title):
        lines = []
        lines.append(f"### {title} ($N = {res['count']}$ runs)\n")
        lines.append("The marginal effect tables show the isolated impact of lookback horizon and resample window on performance by averaging results across all other parameters.\n")
        
        # Marginal Lookback
        lines.append(f"**Marginal Effect: Lookback Horizon ({res['name']})**:\n")
        lines.append("| Lookback Horizon | F1 (Mean ± Std) | F1 Median | AUROC (Mean ± Std) | AUROC Median | Bal Acc (Mean) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for lb in lookbacks:
            s = res['lookback_marginal'][lb]
            lines.append(f"| **{lb}** | {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | {s['f1_med']:.4f} | {s['auroc_mean']:.4f} ± {s['auroc_std']:.4f} | {s['auroc_med']:.4f} | {s['bacc_mean']:.4f} |")
        lines.append("\n")

        # Marginal Window
        lines.append(f"**Marginal Effect: Resample Frequency / Sampling Granularity ({res['name']})**:\n")
        lines.append("| Resample Window | F1 (Mean ± Std) | F1 Median | AUROC (Mean ± Std) | AUROC Median | Bal Acc (Mean) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for win in windows:
            s = res['window_marginal'][win]
            lines.append(f"| **{win}** | {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | {s['f1_med']:.4f} | {s['auroc_mean']:.4f} ± {s['auroc_std']:.4f} | {s['auroc_med']:.4f} | {s['bacc_mean']:.4f} |")
        lines.append("\n")

        # 2D Heatmaps
        lines.append("The 2D interaction heatmaps display performance across all combinations of lookback horizon (rows) and resample window (columns) to show how the two parameters jointly interact.\n")
        
        # Macro F1
        lines.append(f"**2D Temporal Interaction Heatmap: Macro F1 (Rows: Lookback, Cols: Window)**:\n")
        lines.append("| Lookback \\ Window | 4h | 6h | 8h | 12h |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for lb in lookbacks:
            row_str = f"| **{lb}** | " + " | ".join(f"{res['heatmap_f1'][lb][win]:.4f}" if res['heatmap_f1'][lb][win] is not None else "—" for win in windows) + " |"
            lines.append(row_str)
        lines.append("\n")

        # AUROC
        lines.append(f"**2D Temporal Interaction Heatmap: AUROC (Rows: Lookback, Cols: Window)**:\n")
        lines.append("| Lookback \\ Window | 4h | 6h | 8h | 12h |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for lb in lookbacks:
            row_str = f"| **{lb}** | " + " | ".join(f"{res['heatmap_auroc'][lb][win]:.4f}" if res['heatmap_auroc'][lb][win] is not None else "—" for win in windows) + " |"
            lines.append(row_str)
        lines.append("\n")

        return "\n".join(lines)

    md_content = "# Temporal Dynamics Analysis (FLAML XGBoost Feature Ablation)\n\n"
    md_content += format_tables(res_overall, "Overall Temporal Dynamics across All Conditions")
    md_content += "\n---\n\n"
    md_content += format_tables(res_cond0, "Temporal Dynamics for Condition 0 (Pure Sensor Baseline)")
    md_content += "\n---\n\n"
    md_content += format_tables(res_cond4, "Temporal Dynamics for Condition 4 (All Demographics)")

    Path('reports/temporal_dynamics.md').write_text(md_content, encoding='utf-8')
    print("Wrote reports/temporal_dynamics.md successfully.")

if __name__ == '__main__':
    main()
