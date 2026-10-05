"""Best-case targets: does sensing track a label that actually varies within people?

For each target, summarises the sensor-free baseline, XGBoost on raw sensors and
XGBoost on within-person z-scored sensors, under cyclical and subject-wise CV,
with the same checks as identity_decomposition.py.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/target_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_decomposition import boot_ci, centred, shuffled_pooled  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "logs" / "verify_between_person" / "runs"
TARGETS = ["social_connection", "social_stress"]
MODELS = [
    ("Baseline: participant's positive rate (no sensors)", "baseline_user_rate"),
    ("XGBoost, raw sensors + sleep", "xgboost"),
    ("XGBoost, z-scored sensors + sleep", "z_xgboost"),
]


def load(run):
    f = RUNS / run / "pooled_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.stage == "test"].rename(columns={"prob_class_1": "p"})
    return df[["app_user_id", "fold_index", "y_true", "p"]].reset_index(drop=True)


def within_person(df):
    s = [roc_auc_score(g.y_true, g.p) for _, g in df.groupby("app_user_id") if g.y_true.nunique() == 2]
    return np.mean(s), len(s)


def main():
    rng = np.random.default_rng(0)
    pooled = lambda d: roc_auc_score(d.y_true, d.p)
    pc = lambda d: roc_auc_score(d.y_true, centred(d))
    for target in TARGETS:
        print(f"\n### {target}\n")
        print("Cyclical walk-forward\n")
        print("| Model | Pooled AUROC [95% CI] | Person-centred [95% CI] | Within-person (people) | Pooled after shuffle (mean [95%]) | p |")
        print("|---|---|---|---|---|---|")
        for label, suffix in MODELS:
            df = load(f"{target}_{suffix}")
            if df is None:
                print(f"| {label} | (missing) | | | | |")
                continue
            ci, ci_pc = boot_ci(df, pooled, rng), boot_ci(df, pc, rng)
            wp, n = within_person(df)
            sh = shuffled_pooled(df, rng)
            obs = pooled(df)
            p = (np.sum(sh >= obs) + 1) / (len(sh) + 1)
            print(f"| {label} | {obs:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {pc(df):.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | "
                  f"{wp:.2f} ({n}) | {sh.mean():.2f} [{np.percentile(sh, 2.5):.2f}, {np.percentile(sh, 97.5):.2f}] | {p:.3f} |",
                  flush=True)
        print("\nSubject-wise CV (unseen participants, one round of 5 folds)\n")
        print("| Model | AUROC per fold (mean ± SD) | Person-centred [95% CI] | Within-person (people) |")
        print("|---|---|---|---|")
        for label, suffix in MODELS:
            df = load(f"user_cv_{target}_{suffix}")
            if df is None:
                print(f"| {label} | (missing) | | |")
                continue
            folds = [roc_auc_score(g.y_true, g.p) for _, g in df.groupby("fold_index") if g.y_true.nunique() == 2]
            ci_pc = boot_ci(df, pc, rng)
            wp, n = within_person(df)
            print(f"| {label} | {np.mean(folds):.2f} ± {np.std(folds, ddof=1):.2f} | "
                  f"{pc(df):.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | {wp:.2f} ({n}) |", flush=True)


if __name__ == "__main__":
    main()
