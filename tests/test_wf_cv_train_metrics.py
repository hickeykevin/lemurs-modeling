"""Unit tests for wf_cv_train.py's pooled-metric helpers (no DB, no Trainer --
pure functions over a synthetic pooled-predictions DataFrame). See
tests/test_wf_cv_train.py for the real-DB, full-pipeline integration tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import RankedLogger
from src.wf_cv_train import _cluster_bootstrap_ci, _compute_pooled_metrics

log = RankedLogger(__name__, rank_zero_only=True)


def _synthetic_pool(n_users=30, seed=0, positive_user_frac=0.4):
    """A pooled-predictions-shaped DataFrame with scores correlated with
    y_true (so AUROC/AUPRC are meaningfully above chance), positives
    concentrated in a subset of users -- matching this study's real
    cohort shape (see CohortSplitter's class docstring)."""
    rng = np.random.RandomState(seed)
    rows = []
    for uid in range(n_users):
        n = rng.randint(5, 40)
        base_rate = rng.uniform(0.05, 0.5) if rng.rand() < positive_user_frac else 0.0
        for _ in range(n):
            y = int(rng.rand() < base_rate)
            score = float(np.clip(y * 0.6 + rng.normal(0.3, 0.2), 0, 1))
            rows.append({"app_user_id": uid, "y_true": y, "prob_class_0": 1 - score, "prob_class_1": score})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_ci_returns_ordered_bounds_around_point_estimate():
    df = _synthetic_pool()
    from sklearn.metrics import roc_auc_score, average_precision_score

    point_auroc = roc_auc_score(df["y_true"], df["prob_class_1"])
    point_auprc = average_precision_score(df["y_true"], df["prob_class_1"])

    ci = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=300, seed=0, log=log)

    assert ci["pooled/auroc_ci_low"] <= ci["pooled/auroc_ci_high"]
    assert ci["pooled/auprc_ci_low"] <= ci["pooled/auprc_ci_high"]
    # BCa can shift bounds away from the point estimate (that's the whole
    # point of the bias correction), but they shouldn't be wildly off --
    # loosely bound them within [0, 1] and roughly centered near the point.
    assert 0.0 <= ci["pooled/auroc_ci_low"] <= 1.0
    assert 0.0 <= ci["pooled/auroc_ci_high"] <= 1.0
    assert ci["pooled/auroc_ci_low"] <= point_auroc + 0.15
    assert ci["pooled/auroc_ci_high"] >= point_auroc - 0.15


def test_cluster_bootstrap_ci_is_reproducible_with_same_seed():
    df = _synthetic_pool()
    ci1 = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=200, seed=42, log=log)
    ci2 = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=200, seed=42, log=log)
    assert ci1 == ci2


def test_cluster_bootstrap_ci_differs_across_seeds():
    df = _synthetic_pool()
    ci1 = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=200, seed=1, log=log)
    ci2 = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=200, seed=2, log=log)
    assert ci1 != ci2


def test_cluster_bootstrap_ci_single_positive_user_is_degenerate_not_crashing():
    """When only one user ever holds a positive label, every leave-one-user-out
    jackknife sample BCa needs is degenerate (single-class) -- BCa cannot be
    constructed at all, not just noisier. This must surface as nan bounds
    (with a warning), not raise, since a single fold in a real run legitimately
    can have this few positive-holding users in a thin cohort."""
    rng = np.random.RandomState(3)
    rows = []
    for uid in range(15):
        n = rng.randint(5, 12)
        is_positive_user = uid == 0
        for _ in range(n):
            y = 1 if is_positive_user and rng.rand() < 0.5 else 0
            score = float(np.clip(y * 0.5 + rng.normal(0.3, 0.2), 0, 1))
            rows.append({"app_user_id": uid, "y_true": y, "prob_class_0": 1 - score, "prob_class_1": score})
    df = pd.DataFrame(rows)
    assert df.groupby("app_user_id")["y_true"].sum().gt(0).sum() == 1  # exactly one positive-holding user

    ci = _cluster_bootstrap_ci(df, "prob_class_1", n_bootstraps=100, seed=0, log=log)
    assert np.isnan(ci["pooled/auroc_ci_low"])
    assert np.isnan(ci["pooled/auroc_ci_high"])
    assert np.isnan(ci["pooled/auprc_ci_low"])
    assert np.isnan(ci["pooled/auprc_ci_high"])


def test_compute_pooled_metrics_end_to_end_includes_bca_ci_keys():
    df = _synthetic_pool()
    metrics = _compute_pooled_metrics(df, log)

    for key in (
        "pooled/auroc", "pooled/auprc", "pooled/n_predictions", "pooled/n_users",
        "pooled/n_positive", "pooled/auroc_ci_low", "pooled/auroc_ci_high",
        "pooled/auprc_ci_low", "pooled/auprc_ci_high",
    ):
        assert key in metrics
    assert metrics["pooled/auroc_ci_low"] <= metrics["pooled/auroc"] <= metrics["pooled/auroc_ci_high"] \
        or np.isnan(metrics["pooled/auroc_ci_low"])  # BCa CI need not straddle the point estimate exactly


def test_compute_pooled_metrics_empty_pool_returns_empty_dict():
    df = pd.DataFrame(columns=["app_user_id", "y_true", "prob_class_0", "prob_class_1"])
    metrics = _compute_pooled_metrics(df, log)
    assert metrics == {}


def test_compute_pooled_metrics_single_class_returns_empty_dict():
    df = pd.DataFrame({
        "app_user_id": [1, 1, 2, 2],
        "y_true": [0, 0, 0, 0],
        "prob_class_0": [0.6, 0.7, 0.5, 0.9],
        "prob_class_1": [0.4, 0.3, 0.5, 0.1],
    })
    metrics = _compute_pooled_metrics(df, log)
    assert metrics == {}
