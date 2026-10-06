"""Within-person ranking test: can a model that is only ever trained to order a person's own
surveys find any state signal?

XGBoost with ``rank:pairwise`` and the participant as the query group only learns from
pairs *within* a participant (one of their positive surveys vs one of their negative ones),
so recognising who someone is cannot reduce its loss. Participants with a single label
contribute no pairs. It is compared with an ordinary ``binary:logistic`` XGBoost with the
same hyperparameters, so the objective is the only difference.

Data, folds and inputs match the RQ1 FLAML runs: cyclical walk-forward (5 folds, 120 h
purge), dual scaler, ``rolling_72h_6h``, step+distance, hours slept; features are the
flattened window plus the static vector, as ``FLAMLHealthModule`` builds them without a
featurizer.

Reports pooled / person-centred / within-person AUROC (as ``sweep_decomposition.py``) and a
permutation p-value for within-person AUROC: labels shuffled within each test participant
(keeps each person's positive count), 2,000 times.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/within_person_rank.py [sampler] [modalities]
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra import compose, initialize_config_dir
from sklearn.metrics import roc_auc_score

from src.utils.resolvers import register_resolvers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_decomposition import decompose  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SAMPLER = sys.argv[1] if len(sys.argv) > 1 else "rolling_72h_6h"
MODS = sys.argv[2] if len(sys.argv) > 2 else "step_distance"
OUT = REPO / "logs" / "verify_between_person" / "runs" / f"within_person_rank_{SAMPLER}_{MODS}"
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=1, random_state=7, n_jobs=8, tree_method="hist")
OVERRIDES = [
    "eval_plan=cyclical", f"data/sampler=sweep_configs/{SAMPLER}", f"data/modalities={MODS}",
    "data/scaler=dual", "data.os_filter=both", "data.collapse_strategy=none", "data.use_demographics=false",
    "data.use_sleep=true", "data.sleep_feature_mode=hours_only", "data.use_survey_context=false",
    "data.require_sensor_data=true", "data.purge_hours=120.0", "model=flaml_xgboost", "trainer=cpu",
]


def split_arrays(ds):
    """Flattened window + static vector (as FLAMLHealthModule without a featurizer), labels, meta."""
    items = [ds[i] for i in range(len(ds))]
    seq = np.stack([s["features"].numpy().reshape(-1) for s in items])
    parts = [seq]
    if "demographics" in items[0]:
        parts.append(np.stack([s["demographics"].numpy().reshape(-1) for s in items]))
    X = np.concatenate(parts, axis=1).astype(np.float32)
    y = np.array([int(s["targets"]) for s in items])
    links = ds.data_links.reset_index(drop=True)
    return X, y, links["app_user_id"].to_numpy(), links["record_timestamp"].to_numpy()


def within_person_auroc(df, col="p"):
    vals = [roc_auc_score(g.y_true, g[col]) for _, g in df.groupby("app_user_id") if g.y_true.nunique() == 2]
    return float(np.mean(vals)) if vals else np.nan


def permutation_p(df, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = within_person_auroc(df)
    mixed = [g for _, g in df.groupby("app_user_id") if g.y_true.nunique() == 2]
    null = []
    for _ in range(n):
        vals = [roc_auc_score(rng.permutation(g.y_true.to_numpy()), g.p.to_numpy()) for g in mixed]
        null.append(np.mean(vals))
    null = np.array(null)
    return obs, float((np.sum(null >= obs) + 1) / (n + 1)), float(np.percentile(null, 95))


def main():
    register_resolvers()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=OVERRIDES)
    probe = hydra.utils.instantiate(cfg.data)
    probe.setup()
    cohort, n_folds = probe.get_prebuilt_cohort(), probe.get_num_folds()

    rows = []
    for fold in range(n_folds):
        dm = hydra.utils.instantiate(cfg.data, current_fold=fold, prebuilt_cohort=cohort)
        dm.setup()
        Xtr, ytr, utr, _ = split_arrays(dm.data_train)
        Xte, yte, ute, tte = split_arrays(dm.data_test)

        order = np.argsort(utr, kind="stable")  # XGBRanker needs each query group contiguous
        ranker = xgb.XGBRanker(objective="rank:pairwise", **PARAMS)
        ranker.fit(Xtr[order], ytr[order], qid=utr[order])
        w = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)  # as auto_class_weights
        clf = xgb.XGBClassifier(objective="binary:logistic", **PARAMS).fit(Xtr, ytr, sample_weight=w)

        mixed_train = sum(len(np.unique(ytr[utr == u])) == 2 for u in np.unique(utr))
        print(f"fold {fold}: train {len(ytr)} surveys / {len(np.unique(utr))} users ({mixed_train} with both labels); "
              f"test {len(yte)} / {len(np.unique(ute))}", flush=True)
        for name, score in (("rank_within_person", ranker.predict(Xte)), ("binary_classifier", clf.predict_proba(Xte)[:, 1])):
            rows.append(pd.DataFrame({"model": name, "fold_index": fold, "stage": "test", "app_user_id": ute,
                                      "record_timestamp": tte, "y_true": yte, "p": score}))

    pred = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / "pooled_predictions.csv", index=False)

    print(f"\n{SAMPLER}, {MODS}, cyclical 5-fold, {pred.app_user_id.nunique()} participants\n")
    print("| Objective | Pooled | Per fold | Person-rate | Person-centred | Within-person | p (within-person) | 95% of null |")
    print("|---|---|---|---|---|---|---|---|")
    for name, df in pred.groupby("model", sort=False):
        r = decompose(df)
        fold = [roc_auc_score(g.y_true, g.p) for _, g in df.groupby("fold_index") if g.y_true.nunique() == 2]
        obs, p, q95 = permutation_p(df)
        print(f"| {name} | {r['pooled']:.3f} | {np.mean(fold):.3f} ± {np.std(fold):.3f} | {r['person_rate']:.3f} | "
              f"{r['person_centred']:.3f} | {obs:.3f} (n={r['mixed_users']}) | {p:.3f} | {q95:.3f} |")
    print("\nPer participant (within-person AUROC, participants with both labels in test):")
    per = (pred.groupby(["model", "app_user_id"])
           .apply(lambda g: roc_auc_score(g.y_true, g.p) if g.y_true.nunique() == 2 else np.nan, include_groups=False)
           .dropna().unstack(0).round(2))
    print(per.to_string())


if __name__ == "__main__":
    main()
