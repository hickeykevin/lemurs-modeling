"""Population pipeline (log1p + global scaling + has-data mask, optionally past_z) vs the dual-scaler runs.

Same model, target and evaluation scheme in each row pair; only the sampler/scaler differ.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/pop_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_decomposition import boot_ci, centred, shuffled_pooled  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "logs" / "verify_between_person" / "runs"
# (label, [(pipeline name, cyclical run dir)]); subject-wise runs are "user_cv_" + the same dir.
PAIRS = [
    ("suicide_risk, XGBoost", [("dual scaler", "xgboost"), ("log1p + mask", "pop_xgboost"),
                               ("log1p + mask, past_z", "pop_z_xgboost")]),
    ("suicide_risk, Transformer", [("dual scaler", "transformer"), ("log1p + mask", "pop_transformer")]),
    ("social_connection, XGBoost", [("dual scaler", "social_connection_xgboost"),
                                    ("dual scaler, past_z", "social_connection_z_xgboost"),
                                    ("log1p + mask", "social_connection_pop_xgboost"),
                                    ("log1p + mask, past_z", "social_connection_pop_z_xgboost")]),
]


def load(run):
    f = RUNS / run / "pooled_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df[df.stage == "test"].rename(columns={"prob_class_1": "p"}).reset_index(drop=True)


def within_person(df, rng):
    g = [(d.y_true.to_numpy(), d.p.to_numpy()) for _, d in df.groupby("app_user_id") if d.y_true.nunique() == 2]
    auc = np.array([roc_auc_score(y, p) for y, p in g])
    null = np.array([np.mean([roc_auc_score(rng.permutation(y), p) for y, p in g]) for _ in range(1000)])
    return auc.mean(), (np.sum(null >= auc.mean()) + 1) / 1001


def main():
    rng = np.random.default_rng(0)
    pooled = lambda d: roc_auc_score(d.y_true, d.p)
    pc = lambda d: roc_auc_score(d.y_true, centred(d))
    print("### Cyclical walk-forward\n")
    print("| Target, model | Pipeline | Pooled [95% CI] | Person-centred [95% CI] | Within-person (p) | Shuffle p |")
    print("|---|---|---|---|---|---|")
    for label, variants in PAIRS:
        for name, run in variants:
            df = load(run)
            if df is None:
                print(f"| {label} | {name} | (missing) | | | |")
                continue
            ci, ci_pc = boot_ci(df, pooled, rng), boot_ci(df, pc, rng)
            wp, wp_p = within_person(df, rng)
            sh = shuffled_pooled(df, rng)
            p = (np.sum(sh >= pooled(df)) + 1) / (len(sh) + 1)
            print(f"| {label} | {name} | {pooled(df):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                  f"{pc(df):.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | {wp:.2f} ({wp_p:.3f}) | {p:.3f} |", flush=True)
    print("\n### Subject-wise CV (unseen participants)\n")
    print("| Target, model | Pipeline | AUROC per fold (mean ± SD) | Person-centred [95% CI] | Within-person (p) |")
    print("|---|---|---|---|---|")
    for label, variants in PAIRS:
        for name, run in variants:
            df = load(f"user_cv_{run}")
            if df is None:
                print(f"| {label} | {name} | (missing) | | |")
                continue
            folds = [pooled(g) for _, g in df.groupby("fold_index") if g.y_true.nunique() == 2]
            ci_pc = boot_ci(df, pc, rng)
            wp, wp_p = within_person(df, rng)
            print(f"| {label} | {name} | {np.mean(folds):.2f} ± {np.std(folds, ddof=1):.2f} | "
                  f"{pc(df):.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | {wp:.2f} ({wp_p:.3f}) |", flush=True)


if __name__ == "__main__":
    main()
