"""WBM (frozen pretrained encoder) vs the earlier subject-wise runs, suicide risk.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/wbm_summary.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_decomposition import boot_ci, centred  # noqa: E402
from pop_summary import load, within_person  # noqa: E402

ROWS = [
    ("XGBoost, dual scaler, 72h/6h", "user_cv_xgboost"),
    ("XGBoost, log1p + mask, 72h/6h", "user_cv_pop_xgboost"),
    ("Transformer, dual scaler, 72h/6h", "user_cv_transformer"),
    ("Transformer, log1p + mask, 72h/6h", "user_cv_pop_transformer"),
    ("WBM (frozen), unscaled, 168h/1h", "user_cv_wbm"),
    ("WBM fine-tuned, lr 1e-4, 15 epochs", "user_cv_wbm_ft_lr0.0001_r2"),
    ("WBM fine-tuned, lr 1e-5, 15 epochs", "user_cv_wbm_ft_lr0.00001_r2"),
]


def main():
    rng = np.random.default_rng(0)
    pooled = lambda d: roc_auc_score(d.y_true, d.p)
    pc = lambda d: roc_auc_score(d.y_true, centred(d))
    print("| Model | Surveys (positive) | AUROC per fold (mean ± SD) | Fold AUROCs | Person-centred [95% CI] | Within-person (p) |")
    print("|---|---|---|---|---|---|")
    for label, run in ROWS:
        df = load(run)
        if df is None:
            print(f"| {label} | (missing) | | | | |")
            continue
        folds = [pooled(g) for _, g in df.groupby("fold_index") if g.y_true.nunique() == 2]
        ci = boot_ci(df, pc, rng)
        wp, p = within_person(df, rng)
        print(f"| {label} | {len(df)} ({int(df.y_true.sum())}) | {np.mean(folds):.2f} ± {np.std(folds, ddof=1):.2f} | "
              f"{', '.join(f'{f:.2f}' for f in folds)} | {pc(df):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {wp:.2f} ({p:.3f}) |", flush=True)
    d = load("user_cv_wbm")
    if d is not None:
        print(f"\nWBM predictions: mean p = {d.p.mean():.3f}, SD = {d.p.std():.3f}, "
              f"distinct values = {d.p.round(4).nunique()}, base rate = {d.y_true.mean():.3f}")


if __name__ == "__main__":
    main()
