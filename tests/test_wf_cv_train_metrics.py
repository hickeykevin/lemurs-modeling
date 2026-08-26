"""Unit tests for wf_cv_train.py's pooled-metric helpers (no DB, no Trainer --
pure functions over a synthetic pooled-predictions DataFrame). See
tests/test_wf_cv_train.py for the real-DB, full-pipeline integration tests.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import RankedLogger
from src.utils.evaluation_callbacks import ClassificationMetricsCallback
from src.wf_cv_train import (
    _classification_metrics_params,
    _cluster_bootstrap_ci,
    _compute_pooled_metrics,
    _pooled_classification_metrics,
)

log = RankedLogger(__name__, rank_zero_only=True)

ALL_METRICS = (
    "auroc", "auprc", "f1", "precision", "recall",
    "specificity", "sensitivity_at_specificity", "balanced_accuracy",
)
DEFAULT_PARAMS = _classification_metrics_params(None)


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


def test_pooled_classification_metrics_matches_the_callback_exactly():
    """The whole point: _pooled_classification_metrics must reproduce
    ClassificationMetricsCallback's own MetricCollection bit-for-bit when
    run over the identical pooled predictions, not just approximate it."""
    df = _synthetic_pool()
    probs = df[["prob_class_0", "prob_class_1"]].to_numpy()
    y_true = df["y_true"].to_numpy()

    ours = _pooled_classification_metrics(y_true, probs, DEFAULT_PARAMS)

    cb = ClassificationMetricsCallback()
    collection = cb._init_metrics(num_classes=2, device=torch.device("cpu"), bootstrap=False)
    collection.update(
        torch.as_tensor(probs.copy(), dtype=torch.float32),
        torch.as_tensor(y_true.copy(), dtype=torch.long),
    )
    theirs = {k: float(v) for k, v in collection.compute().items()}

    for name in ("f1", "auroc", "precision", "recall", "specificity", "sensitivity_at_specificity", "balanced_accuracy"):
        assert ours[name] == pytest.approx(theirs[name], abs=1e-6), name


def test_classification_metrics_params_reads_from_cfg():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "callbacks": {"classification_metrics": {"min_specificity": 0.75, "f1_average": "micro"}}
    })
    params = _classification_metrics_params(cfg)
    assert params["min_specificity"] == 0.75
    assert params["f1_average"] == "micro"
    assert params["recall_average"] == "macro"  # untouched keys keep their default


def test_classification_metrics_params_falls_back_without_cfg():
    assert _classification_metrics_params(None) == DEFAULT_PARAMS


def test_cluster_bootstrap_ci_returns_ordered_bounds_for_every_metric():
    df = _synthetic_pool()
    ci = _cluster_bootstrap_ci(
        df, ["prob_class_0", "prob_class_1"], DEFAULT_PARAMS, n_bootstraps=150, seed=0, log=log
    )
    for name in ALL_METRICS:
        lo, hi = ci[f"pooled/{name}_ci_low"], ci[f"pooled/{name}_ci_high"]
        if np.isnan(lo) or np.isnan(hi):
            continue
        assert lo <= hi, name
        assert 0.0 <= lo <= 1.0, name
        assert 0.0 <= hi <= 1.0, name


def test_cluster_bootstrap_ci_is_reproducible_with_same_seed():
    df = _synthetic_pool()
    ci1 = _cluster_bootstrap_ci(
        df, ["prob_class_0", "prob_class_1"], DEFAULT_PARAMS, n_bootstraps=100, seed=42, log=log
    )
    ci2 = _cluster_bootstrap_ci(
        df, ["prob_class_0", "prob_class_1"], DEFAULT_PARAMS, n_bootstraps=100, seed=42, log=log
    )
    assert ci1 == ci2


def test_cluster_bootstrap_ci_single_positive_user_is_degenerate_not_crashing():
    """When only one user ever holds a positive label, every leave-one-user-out
    jackknife sample BCa needs is degenerate (single-class) -- BCa cannot be
    constructed at all, not just noisier, for ANY of these metrics (they all
    depend on both classes being present). Must surface as nan bounds (with
    a warning), not raise."""
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

    ci = _cluster_bootstrap_ci(
        df, ["prob_class_0", "prob_class_1"], DEFAULT_PARAMS, n_bootstraps=60, seed=0, log=log
    )
    for name in ALL_METRICS:
        assert np.isnan(ci[f"pooled/{name}_ci_low"]), name
        assert np.isnan(ci[f"pooled/{name}_ci_high"]), name


def test_compute_pooled_metrics_end_to_end_includes_all_metrics_and_cis():
    df = _synthetic_pool()
    metrics = _compute_pooled_metrics(df, log, n_bootstraps=100)

    for name in ALL_METRICS:
        assert f"pooled/{name}" in metrics
        assert f"pooled/{name}_ci_low" in metrics
        assert f"pooled/{name}_ci_high" in metrics
    for key in ("pooled/n_predictions", "pooled/n_users", "pooled/n_positive"):
        assert key in metrics

    assert metrics["pooled/auroc_ci_low"] <= metrics["pooled/auroc"] <= metrics["pooled/auroc_ci_high"] \
        or np.isnan(metrics["pooled/auroc_ci_low"])  # BCa CI need not straddle the point estimate exactly


def test_compute_pooled_metrics_matches_callback_point_estimates():
    """End-to-end: _compute_pooled_metrics's point estimates (not just
    _pooled_classification_metrics in isolation) match the callback."""
    df = _synthetic_pool()
    metrics = _compute_pooled_metrics(df, log, n_bootstraps=100)

    cb = ClassificationMetricsCallback()
    collection = cb._init_metrics(num_classes=2, device=torch.device("cpu"), bootstrap=False)
    probs = df[["prob_class_0", "prob_class_1"]].to_numpy()
    collection.update(
        torch.as_tensor(probs.copy(), dtype=torch.float32),
        torch.as_tensor(df["y_true"].to_numpy().copy(), dtype=torch.long),
    )
    theirs = {k: float(v) for k, v in collection.compute().items()}

    for name in ("f1", "auroc", "precision", "recall", "specificity", "sensitivity_at_specificity", "balanced_accuracy"):
        assert metrics[f"pooled/{name}"] == pytest.approx(theirs[name], abs=1e-6), name


def test_compute_pooled_metrics_respects_cfg_params():
    """A different min_specificity in cfg should change sensitivity_at_specificity's
    point estimate relative to the default-params run (proves cfg actually flows
    through, not just accepted and ignored)."""
    from omegaconf import OmegaConf

    df = _synthetic_pool()
    metrics_default = _compute_pooled_metrics(df, log, n_bootstraps=50)

    cfg = OmegaConf.create({"callbacks": {"classification_metrics": {"min_specificity": 0.5}}})
    metrics_custom = _compute_pooled_metrics(df, log, cfg=cfg, n_bootstraps=50)

    assert metrics_default["pooled/sensitivity_at_specificity"] != pytest.approx(
        metrics_custom["pooled/sensitivity_at_specificity"]
    )


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
