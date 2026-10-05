"""Is a run's pooled AUROC identity, or within-person tracking?

For each cyclical run, holding its predictions fixed:
  * pooled AUROC with a participant-clustered bootstrap 95% CI
  * person-centred AUROC (each person's mean prediction removed) with CI
  * within-person label shuffle (Chaibub Neto et al., 2019): permute labels
    among each person's own surveys, keeping each person's positive rate.
    If the observed pooled AUROC sits inside the shuffled distribution, the
    pooled score is fully explained by who the survey belongs to; the
    one-sided p is the share of shuffles at or above the observed value.

Usage (compute node):
    .venv/bin/python scripts/verify_between_person/identity_decomposition.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "logs" / "verify_between_person" / "runs"
N_SHUFFLE, N_BOOT, SEED = 1000, 1000, 0

CYCLICAL = [
    ("Baseline: participant's positive rate (no sensors)", "baseline_user_rate"),
    ("Sleep only", "abl_sleep_only"),
    ("Raw sensors + sleep (A)", "xgboost"),
    ("Raw sensors, no sleep", "abl_raw_nosleep"),
    ("Centred sensors + sleep (D)", "centered_xgboost"),
    ("Centred sensors, no sleep", "abl_centered_nosleep"),
    ("z-scored sensors + sleep", "abl_z_sleep"),
    ("z-scored sensors, no sleep", "abl_z_nosleep"),
]


def load(run):
    f = RUNS / run / "pooled_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.stage == "test"].rename(columns={"prob_class_1": "p"})
    return df[["app_user_id", "y_true", "p"]].reset_index(drop=True)


def centred(df):
    return df.p - df.groupby("app_user_id").p.transform("mean")


def boot_ci(df, stat, rng):
    users = df.app_user_id.unique()
    idx = {u: np.flatnonzero(df.app_user_id.to_numpy() == u) for u in users}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(users, size=len(users), replace=True)
        rows = np.concatenate([idx[u] for u in pick])
        d = df.iloc[rows].copy()
        # Duplicated people become distinct people for the centring step.
        d["app_user_id"] = np.repeat(np.arange(len(pick)), [len(idx[u]) for u in pick])
        if d.y_true.nunique() == 2:
            vals.append(stat(d))
    return np.percentile(vals, [2.5, 97.5])


def shuffled_pooled(df, rng):
    y, p, g = df.y_true.to_numpy(), df.p.to_numpy(), df.app_user_id.to_numpy()
    groups = [np.flatnonzero(g == u) for u in np.unique(g)]
    out = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        ys = y.copy()
        for ix in groups:
            ys[ix] = rng.permutation(y[ix])
        out[i] = roc_auc_score(ys, p)
    return out


def main():
    rng = np.random.default_rng(SEED)
    pooled = lambda d: roc_auc_score(d.y_true, d.p)
    pc = lambda d: roc_auc_score(d.y_true, centred(d))
    print("| Model | Pooled AUROC [95% CI] | Person-centred AUROC [95% CI] | Pooled after within-person shuffle (mean [95%]) | Lift kept after shuffle | p (observed ≤ shuffled) |")
    print("|---|---|---|---|---|---|")
    for label, run in CYCLICAL:
        df = load(run)
        if df is None:
            print(f"| {label} | (missing) | | | | |")
            continue
        obs, obs_pc = pooled(df), pc(df)
        ci, ci_pc = boot_ci(df, pooled, rng), boot_ci(df, pc, rng)
        sh = shuffled_pooled(df, rng)
        kept = (sh.mean() - 0.5) / (obs - 0.5) if obs != 0.5 else np.nan
        p = (np.sum(sh >= obs) + 1) / (N_SHUFFLE + 1)
        print(
            f"| {label} | {obs:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {obs_pc:.2f} [{ci_pc[0]:.2f}, {ci_pc[1]:.2f}] | "
            f"{sh.mean():.2f} [{np.percentile(sh, 2.5):.2f}, {np.percentile(sh, 97.5):.2f}] | {kept:.0%} | {p:.3f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
