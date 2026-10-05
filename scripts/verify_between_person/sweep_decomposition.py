"""Identity decomposition across every saved run of the cyclical sweep.

For each run under logs/cyclical_sweep/runs that saved pooled_predictions.csv:
  * pooled AUROC (what the sweep reports)
  * person-rate AUROC: scoring each survey with the participant's positive rate in their
    OTHER folds -- a sensor-free baseline built from the run's own labels
  * within-person AUROC (mean over participants with both states) and its person-centred
    counterpart (each person's mean prediction removed)
Settings are read from the run's .hydra/config.yaml. Writes a per-run CSV and prints a
markdown summary.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/sweep_decomposition.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "logs" / "cyclical_sweep" / "runs"
OUT = REPO / "reports" / "sweep_identity_decomposition.csv"


def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def meta(d: Path) -> dict | None:
    try:
        c = yaml.safe_load(open(d / ".hydra" / "config.yaml"))
    except (OSError, yaml.YAMLError):
        return None
    data, model = c.get("data", {}), c.get("model", {})
    net = model.get("net") if isinstance(model.get("net"), dict) else {}
    est = (model.get("automl_config") or {}).get("estimator_list")
    feat = model.get("featurizer") if isinstance(model.get("featurizer"), dict) else {}
    scaler = data.get("scaler")
    return dict(
        model=est[0] if est else net.get("_target_", "").rsplit(".", 1)[-1],
        featurizer=feat.get("_target_", "none").rsplit(".", 1)[-1],
        scaler=(scaler.get("_target_", "?").rsplit(".", 1)[-1] if isinstance(scaler, dict) else "none"),
        lookback=data.get("sampler", {}).get("lookback_hours"),
        freq=data.get("sampler", {}).get("resample_freq"),
        modalities="+".join(data.get("modalities", [])),
        use_sleep=bool(data.get("use_sleep")),
    )


def decompose(df: pd.DataFrame) -> dict:
    y, p = df.y_true.to_numpy(), df.p.to_numpy()
    g = df.groupby("app_user_id")
    centred = p - g.p.transform("mean").to_numpy()
    # person's positive rate in their other folds (falls back to the overall rate)
    tot, n = g.y_true.transform("sum"), g.y_true.transform("size")
    ft, fn = df.groupby(["app_user_id", "fold_index"]).y_true.transform("sum"), df.groupby(["app_user_id", "fold_index"]).y_true.transform("size")
    rate = np.where(n - fn > 0, (tot - ft) / (n - fn).clip(lower=1), y.mean())
    wp = [auroc(x.y_true, x.p) for _, x in g if x.y_true.nunique() == 2]
    return dict(
        surveys=len(df), users=df.app_user_id.nunique(), mixed_users=len(wp),
        pooled=auroc(y, p), person_rate=auroc(y, rate), person_centred=auroc(y, centred),
        within_person=float(np.mean(wp)) if wp else np.nan,
    )


def main():
    rows = []
    for d in sorted(ROOT.glob("*/")):
        f = d / "pooled_predictions.csv"
        m = meta(d) if f.exists() else None
        if m is None:
            continue
        df = pd.read_csv(f)
        df = df[df.stage == "test"].rename(columns={"prob_class_1": "p"})
        if df.y_true.nunique() < 2:
            continue
        rows.append({"run": d.name, **m, **decompose(df)})
    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)

    q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]"
    print(f"{len(r)} runs ({r.run.min()} to {r.run.max()}); surveys per run {r.surveys.min()}-{r.surveys.max()}, "
          f"participants {r.users.min()}-{r.users.max()}, with both states {r.mixed_users.min()}-{r.mixed_users.max()}")
    print("scalers:", r.scaler.value_counts().to_dict(), "| use_sleep:", r.use_sleep.value_counts().to_dict())
    print("\n| Quantity | Median [IQR] | Range |\n|---|---|---|")
    for lab, c in (("Pooled AUROC (what the sweep reports)", "pooled"), ("Person-rate AUROC (no sensors)", "person_rate"),
                   ("Person-centred AUROC", "person_centred"), ("Within-person AUROC", "within_person")):
        print(f"| {lab} | {q(r[c])} | {r[c].min():.2f}–{r[c].max():.2f} |")
    print(f"\nRuns whose pooled AUROC is below the sensor-free person-rate AUROC: {(r.pooled < r.person_rate).mean():.1%}")
    print(f"Runs with within-person AUROC above 0.55: {(r.within_person > .55).mean():.1%}; above 0.60: {(r.within_person > .60).mean():.1%}")
    print(f"Runs with person-centred AUROC above 0.55: {(r.person_centred > .55).mean():.1%}")
    for a, b in (("pooled", "within_person"), ("pooled", "person_centred"), ("pooled", "person_rate")):
        print(f"Spearman({a}, {b}) = {spearmanr(r[a], r[b], nan_policy='omit')[0]:.2f}")
    print("\nBy model (median):\n\n| Model | Runs | Pooled | Person-rate | Person-centred | Within-person |\n|---|---|---|---|---|---|")
    for k, g in r.groupby("model"):
        print(f"| {k} | {len(g)} | {g.pooled.median():.2f} | {g.person_rate.median():.2f} | {g.person_centred.median():.2f} | {g.within_person.median():.2f} |")
    print("\nBy modalities (median):\n\n| Modalities | Runs | Pooled | Person-centred | Within-person |\n|---|---|---|---|---|")
    for k, g in r.groupby("modalities"):
        print(f"| {k} | {len(g)} | {g.pooled.median():.2f} | {g.person_centred.median():.2f} | {g.within_person.median():.2f} |")


if __name__ == "__main__":
    main()
