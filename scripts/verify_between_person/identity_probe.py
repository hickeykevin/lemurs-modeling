"""Identity probe: how well does each feature group tell participants apart?

Builds the same cohort and features as the cyclical XGBoost runs (fold 0's
train/val/test pooled, i.e. every survey not purged at that fold), then trains a
classifier to predict *which participant* a survey belongs to from one feature
group at a time. Record-wise 5-fold CV is deliberate here: the question is how
recognisable a person is from these features, not whether anything generalises.

Groups:
  sleep           -- previous-night sleep hours (the uncentred static feature)
  raw             -- flattened 72h x 6h sensor window, as the A runs see it
  past_mean       -- raw minus the user's mean over earlier surveys (D runs)
  |past_mean|     -- magnitude only: tests "how much a person's deviations vary"
  past_z          -- past_mean divided by the user's earlier-survey SD
  past_z + sleep  -- what the abl_z_sleep run sees
Each group is also scored with participant IDs permuted (the chance level).

Metric: macro one-vs-rest AUROC over participants (chance 0.5) and top-1
accuracy (chance = largest participant's share).

Usage (compute node):
    .venv/bin/python scripts/verify_between_person/identity_probe.py
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models.health_module import past_centre
from src.utils.resolvers import register_resolvers

REPO = Path(__file__).resolve().parents[2]
OVERRIDES = [
    "eval_plan=cyclical", "data/sampler=sweep_configs/rolling_72h_6h",
    "data/modalities=step_calorie_distance", "data/scaler=dual", "data.os_filter=both",
    "data.collapse_strategy=none", "data.use_demographics=false", "data.use_sleep=true",
    "data.sleep_feature_mode=hours_only", "data.use_survey_context=false",
    "data.require_sensor_data=true", "data.purge_hours=120.0",
]
SEED = 0


def load_features():
    register_resolvers()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=OVERRIDES)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    seqs, sleep, links = [], [], []
    for ds in (dm.data_train, dm.data_val, dm.data_test):
        if ds is None or len(ds) == 0:
            continue
        seqs.append(ds._sequences.reshape(len(ds), -1).astype(np.float64))
        sleep.append(ds.sleep_features.astype(np.float64))
        links.append(ds.data_links[["app_user_id", "record_timestamp"]])
    links = pd.concat(links, ignore_index=True)
    shape = ds._sequences.shape[1:]  # (time steps, channels)
    return np.concatenate(seqs), np.concatenate(sleep), links, shape, dm.hparams.sampler.num_time_features


def probe(X, y, rng):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    return (
        roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes),
        accuracy_score(y, classes[proba.argmax(1)]),
    )


def main():
    raw, sleep, links, (T, C), n_time = load_features()
    users = links.app_user_id.to_numpy()
    times = pd.to_datetime(links.record_timestamp).to_numpy()
    # Participants need >= 5 surveys for 5-fold stratification.
    keep = pd.Series(users).map(pd.Series(users).value_counts()).to_numpy() >= 5
    centred = past_centre(raw, users, times, "past_mean")
    z = past_centre(raw, users, times, "past_z")
    groups = {
        "sleep": sleep,
        "raw": raw,
        "past_mean": centred,
        "|past_mean|": np.abs(centred),
        "past_z": z,
        "past_z + sleep": np.c_[z, sleep],
    }
    # Channel breakdown. The window is [T, C] flattened row-major. With the dual
    # scaler, channels are the modalities scaled globally, then the same
    # modalities scaled per subject (sampler time channels, if any, come last).
    z3 = z.reshape(len(z), T, C)
    n_mod = (C - n_time) // 2
    names = sorted(["step", "calorie", "distance"])[:n_mod]  # HealthDataset sorts modalities
    glob, subj = list(range(n_mod)), list(range(n_mod, 2 * n_mod))
    groups["past_z, globally scaled channels"] = z3[:, :, glob].reshape(len(z), -1)
    groups["past_z, subject-scaled channels"] = z3[:, :, subj].reshape(len(z), -1)
    for i, c in enumerate(glob):
        groups[f"past_z, {names[i]} (global)"] = z3[:, :, c]
    y = users[keep]
    rng = np.random.default_rng(SEED)
    chance_acc = pd.Series(y).value_counts(normalize=True).iloc[0]
    print(f"window {T} steps x {C} channels ({n_time} time channels last)")
    print(f"{keep.sum()} surveys, {len(np.unique(y))} participants; top-1 chance = {chance_acc:.3f}\n")
    rows = []
    for name, X in groups.items():
        auc, acc = probe(X[keep], y, rng)
        auc_p, acc_p = probe(X[keep], rng.permutation(y), rng)
        rows.append((name, X.shape[1], auc, acc, auc_p, acc_p))
        print(f"done {name}", flush=True)
    print("\n| Features | Dims | Identity AUROC | Top-1 accuracy | AUROC, IDs permuted | Accuracy, IDs permuted |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} |")


if __name__ == "__main__":
    main()
