"""Markdown tables for the between- vs within-person verification runs.

Reads `pooled_predictions.csv` from each run under logs/verify_between_person/runs/
and prints one table per evaluation scheme. Besides pooled / within-person /
person-centred AUROC, it splits surveys by the user's *previous* survey label:

* onset:       previous survey negative -> can the model catch a switch into risk?
* persistence: previous survey positive -> will the risk continue?

Within each subset the previous label is constant, so it cannot do the work.
Previous labels are read from any run built with `data/prev_labels=...` and joined
to the other runs on (app_user_id, record_timestamp); runs without them still get
the subsets.

Usage (on a compute node):
    .venv/bin/python scripts/verify_between_person/summarize_runs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "logs" / "verify_between_person" / "runs"

# (label, run directory) in display order; missing runs are skipped.
CYCLICAL = [
    ("Baseline: participant's positive rate (no sensors)", "baseline_user_rate"),
    ("A. XGBoost, sensors + sleep", "xgboost"),
    ("A. Transformer, sensors + sleep", "transformer"),
    ("B. XGBoost, previous labels + sleep (no sensors)", "prev_only_xgboost"),
    ("C. XGBoost, sensors + sleep + previous labels", "sensors_prev_xgboost"),
    ("C. Transformer, sensors + sleep + previous labels", "sensors_prev_transformer"),
    ("D. XGBoost, within-person-centred sensors + sleep", "centered_xgboost"),
    ("D. XGBoost, within-person-centred sensors + sleep + previous labels", "centered_sensors_prev_xgboost"),
]
SUBJECT_WISE = [(label, f"user_cv_{run}") for label, run in CYCLICAL]


def load(run: str) -> pd.DataFrame | None:
    f = RUNS / run / "pooled_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.stage == "test"].rename(columns={"prob_class_1": "p"})
    df["record_timestamp"] = pd.to_datetime(df["record_timestamp"])
    return df


def prev_lookup(runs) -> pd.DataFrame | None:
    """(user, timestamp) -> previous risk label, from any run that saved previous labels."""
    for _, run in runs:
        df = load(run)
        if df is not None and "prev_suicide_ideation" in df:
            out = df[["app_user_id", "record_timestamp"]].copy()
            out["prev_risk"] = np.maximum(df.prev_suicide_ideation, df.prev_suicide_preparation)
            out.loc[df.prev_label_missing == 1, "prev_risk"] = np.nan
            return out.drop_duplicates(["app_user_id", "record_timestamp"])
    return None


def auroc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def within_person(df: pd.DataFrame) -> tuple[float, int]:
    s = [auroc(g.y_true, g.p) for _, g in df.groupby("app_user_id") if g.y_true.nunique() == 2]
    return (float(np.mean(s)) if s else np.nan), len(s)


def person_centred(df: pd.DataFrame) -> float:
    return auroc(df.y_true, df.p - df.groupby("app_user_id").p.transform("mean"))


def per_fold(df: pd.DataFrame, subset=None) -> tuple[float, float]:
    d = df if subset is None else df[subset]
    s = [auroc(g.y_true, g.p) for _, g in d.groupby("fold_index")]
    s = [v for v in s if np.isfinite(v)]
    return (float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else np.nan) if s else (np.nan, np.nan)


def fmt(v, digits=2) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def table(runs, scheme: str) -> str:
    lookup = prev_lookup(runs)
    rows = []
    for label, run in runs:
        df = load(run)
        if df is None:
            continue
        if lookup is not None:
            df = df.merge(lookup, on=["app_user_id", "record_timestamp"], how="left")
        onset = df.prev_risk == 0 if "prev_risk" in df else None
        persist = df.prev_risk == 1 if "prev_risk" in df else None
        wp, n_wp = within_person(df)
        if scheme == "cyclical":
            main = fmt(auroc(df.y_true, df.p))
            on = fmt(auroc(df[onset].y_true, df[onset].p)) if onset is not None else "—"
            pe = fmt(auroc(df[persist].y_true, df[persist].p)) if persist is not None else "—"
        else:
            m, sd = per_fold(df)
            main = f"{fmt(m)} ± {fmt(sd)}"
            on = fmt(per_fold(df, onset)[0]) if onset is not None else "—"
            pe = fmt(per_fold(df, persist)[0]) if persist is not None else "—"
        rows.append((label, main, fmt(wp), fmt(person_centred(df)), on, pe, len(df)))

    head_main = "Pooled AUROC" if scheme == "cyclical" else "AUROC per fold (mean ± SD)"
    lines = [
        f"| Model | {head_main} | Within-person | Person-centred | Onset (prev. negative) | Persistence (prev. positive) | Surveys |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]

    # Reference: predict the previous survey's label itself (the `lag` idea), same surveys.
    if lookup is not None and rows:
        df = load(runs[0][1]).merge(lookup, on=["app_user_id", "record_timestamp"], how="left").dropna(subset=["prev_risk"])
        df = df.assign(p=df.prev_risk)
        main = fmt(auroc(df.y_true, df.p)) if scheme == "cyclical" else f"{fmt(per_fold(df)[0])} ± {fmt(per_fold(df)[1])}"
        lines.append(f"| Reference: previous survey's label (no model) | {main} | {fmt(within_person(df)[0])} | "
                     f"{fmt(person_centred(df))} | 0.50 by construction | 0.50 by construction | {len(df)} |")
        counts = (
            load(runs[0][1]).merge(lookup, on=["app_user_id", "record_timestamp"], how="left")
            .groupby("prev_risk").y_true.agg(["size", "sum"])
        )
        notes = ", ".join(f"previous {'positive' if k == 1 else 'negative'}: {int(r['size'])} surveys, {int(r['sum'])} positive"
                          for k, r in counts.iterrows())
        lines.append(f"\nSubset sizes: {notes}.")
    return "\n".join(lines)


if __name__ == "__main__":
    print("### Cyclical walk-forward (within-person temporal)\n")
    print(table(CYCLICAL, "cyclical"))
    print("\n### Subject-wise CV (unseen participants, one round of 5 folds)\n")
    print(table(SUBJECT_WISE, "subject_wise"))
    sys.exit(0)
