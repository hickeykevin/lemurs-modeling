"""Score item-history runs with the identity controls.

For every run dir under logs/item_prediction/runs that saved pooled_predictions.csv:
  * pooled AUROC (pooled over participants; reflects who is who as much as state)
  * person-rate AUROC: each survey scored by the participant's positive rate in their OTHER
    folds -- the sensor-free, item-free identity baseline (not available under user_cv, where
    a participant's other surveys are never seen, so it is a leak-free reference only for cyclical)
  * person-centred AUROC: each person's mean prediction removed
  * within-person AUROC: mean over participants with both states
  * fold AUROC mean +- SD (per-fold AUROC, one number per held-out set)
A participant-clustered bootstrap gives a 95% CI for the pooled and person-centred AUROC.

Usage: PYTHONPATH=. .venv/bin/python scripts/item_prediction/analyze.py [runs_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "logs" / "item_prediction" / "runs"
rng = np.random.default_rng(0)


def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def centred(df):
    return df.p - df.groupby("app_user_id").p.transform("mean")


def person_rate(df):
    g = df.groupby("app_user_id")
    tot, n = g.y_true.transform("sum"), g.y_true.transform("size")
    gf = df.groupby(["app_user_id", "fold_index"])
    ft, fn = gf.y_true.transform("sum"), gf.y_true.transform("size")
    return np.where(n - fn > 0, (tot - ft) / (n - fn).clip(lower=1), df.y_true.mean())


def boot(df, score, n=500):
    users = df.app_user_id.unique()
    by = {u: d for u, d in df.groupby("app_user_id")}
    out = []
    for _ in range(n):
        s = pd.concat([by[u] for u in rng.choice(users, len(users))])
        out.append(score(s))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    rows = []
    for d in sorted(ROOT.glob("*/")):
        f = d / "pooled_predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df.stage == "test"].rename(columns={"prob_class_1": "p"}).reset_index(drop=True)
        wp = [auroc(x.y_true, x.p) for _, x in df.groupby("app_user_id") if x.y_true.nunique() == 2]
        pooled_ci = boot(df, lambda s: auroc(s.y_true, s.p))
        cen_ci = boot(df, lambda s: auroc(s.y_true, centred(s)))
        fold = [auroc(x.y_true, x.p) for _, x in df.groupby("fold_index")]
        rows.append(dict(
            run=d.name, surveys=len(df), users=df.app_user_id.nunique(), pos_rate=df.y_true.mean(),
            pooled=auroc(df.y_true, df.p), pooled_ci=f"{pooled_ci[0]:.2f}-{pooled_ci[1]:.2f}",
            person_rate=auroc(df.y_true, person_rate(df)),
            person_centred=auroc(df.y_true, centred(df)), centred_ci=f"{cen_ci[0]:.2f}-{cen_ci[1]:.2f}",
            within_person=float(np.mean(wp)) if wp else np.nan, mixed_users=len(wp),
            fold_mean=np.nanmean(fold), fold_sd=np.nanstd(fold),
        ))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT.parent / "summary.csv", index=False)
    print("| Run | Surveys | Users | Pos rate | Pooled [95% CI] | Person-rate | Person-centred [95% CI] | Within-person | Fold mean ± SD |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, x in r.iterrows():
        print(f"| {x.run} | {x.surveys} | {x.users} | {x.pos_rate:.2f} | {x.pooled:.2f} [{x.pooled_ci}] | {x.person_rate:.2f} | "
              f"{x.person_centred:.2f} [{x.centred_ci}] | {x.within_person:.2f} (n={x.mixed_users}) | {x.fold_mean:.2f} ± {x.fold_sd:.2f} |")


if __name__ == "__main__":
    main()
