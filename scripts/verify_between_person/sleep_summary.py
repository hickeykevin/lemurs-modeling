"""Sleep features (no sensors): hours vs within-person timing deviations.

Rows compare the old sleep feature (hours, morning surveys only), hours carried to
the same day's afternoon survey, within-person timing deviations, and deviations
plus the previous survey's labels against previous labels + hours (model B).
Onset = surveys following a negative survey (previous labels read from any run
that saved them), the subset where a sleep disruption would matter most.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/sleep_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_decomposition import boot_ci, centred  # noqa: E402
from pop_summary import load, within_person  # noqa: E402

ROWS = [
    ("Hours, morning surveys only (old)", "abl_sleep_only"),
    ("Hours, carried to afternoon", "sleep_hours_carry"),
    ("Timing deviations (within-person)", "sleep_dev"),
    ("Previous labels + hours (B)", "prev_only_xgboost"),
    ("Previous labels + timing deviations", "sleep_dev_prev"),
]


def prev_lookup(runs):
    for run in runs:
        for r in (run, f"user_cv_{run}"):
            df = load(r)
            if df is not None and "prev_suicide_ideation" in df:
                out = df[["app_user_id", "record_timestamp"]].copy()
                out["prev_risk"] = np.maximum(df.prev_suicide_ideation, df.prev_suicide_preparation)
                out.loc[df.prev_label_missing == 1, "prev_risk"] = np.nan
                return out.drop_duplicates(["app_user_id", "record_timestamp"])
    return None


def main():
    rng = np.random.default_rng(0)
    pooled = lambda d: roc_auc_score(d.y_true, d.p)
    pc = lambda d: roc_auc_score(d.y_true, centred(d))
    lookup = prev_lookup([r for _, r in ROWS])

    def onset(df, by_fold):
        d = df.merge(lookup, on=["app_user_id", "record_timestamp"], how="left")
        d = d[d.prev_risk == 0]
        if by_fold:
            s = [pooled(g) for _, g in d.groupby("fold_index") if g.y_true.nunique() == 2]
            return f"{np.mean(s):.2f}"
        return f"{pooled(d):.2f}"

    for scheme, prefix in (("Cyclical walk-forward", ""), ("Subject-wise CV (unseen participants)", "user_cv_")):
        by_fold = bool(prefix)
        print(f"\n### {scheme}\n")
        main_col = "AUROC per fold (mean ± SD)" if by_fold else "Pooled [95% CI]"
        print(f"| Inputs | {main_col} | Person-centred [95% CI] | Within-person (p) | Onset (prev. negative) |")
        print("|---|---|---|---|---|")
        for label, run in ROWS:
            df = load(prefix + run)
            if df is None:
                print(f"| {label} | (not run) | | | |")
                continue
            if by_fold:
                s = [pooled(g) for _, g in df.groupby("fold_index") if g.y_true.nunique() == 2]
                main = f"{np.mean(s):.2f} ± {np.std(s, ddof=1):.2f}"
            else:
                ci = boot_ci(df, pooled, rng)
                main = f"{pooled(df):.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
            ci_pc = boot_ci(df, pc, rng)
            wp, wp_p = within_person(df, rng)
            print(f"| {label} | {main} | {pc(df):.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | "
                  f"{wp:.2f} ({wp_p:.3f}) | {onset(df, by_fold) if lookup is not None else '—'} |", flush=True)


if __name__ == "__main__":
    main()
