"""Tests for src/eval_plans/cv_aggregation.py."""


import math

import numpy as np
import pytest

from src.eval_plans.cv_aggregation import aggregate_cv_metrics
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

OFFSET = 1_000_000


class _RecordingLogger:
    """Captures log_metrics calls so the summary row can be asserted on."""

    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((dict(metrics), step))


def test_mean_std_and_ci_for_a_simple_series():
    metrics = {"test/auroc": [0.8, 0.6, 0.7]}
    agg = aggregate_cv_metrics(metrics, [], log, total_runs=3, log_step_offset=OFFSET)

    values = np.asarray([0.8, 0.6, 0.7])
    assert agg["test/auroc_mean"] == pytest.approx(values.mean())
    # Sample sd (ddof=1), not population sd -- these runs are a sample of splits.
    assert agg["test/auroc_std"] == pytest.approx(values.std(ddof=1))
    assert agg["test/auroc_n_runs"] == 3.0

    stderr = values.std(ddof=1) / math.sqrt(3)
    assert agg["test/auroc_ci_low"] == pytest.approx(values.mean() - 1.96 * stderr)
    assert agg["test/auroc_ci_high"] == pytest.approx(values.mean() + 1.96 * stderr)


def test_emits_both_bare_and_cv_summary_prefixed_keys():
    """Both key sets are part of the contract: bare keys feed
    get_metric_value/hparams_search, cv_summary/ keys are what gets logged."""
    agg = aggregate_cv_metrics({"test/f1": [0.5, 0.7]}, [], log, 2, OFFSET)
    for suffix in ("mean", "std", "ci_low", "ci_high", "n_runs"):
        assert f"test/f1_{suffix}" in agg
        assert f"cv_summary/test/f1_{suffix}" in agg
        assert agg[f"test/f1_{suffix}"] == agg[f"cv_summary/test/f1_{suffix}"]


def test_nans_are_excluded_not_propagated():
    """A fold whose test users are all one class yields an undefined AUROC.
    That must not erase the folds that were defined."""
    agg = aggregate_cv_metrics({"test/auroc": [0.8, float("nan"), 0.6]}, [], log, 3, OFFSET)
    assert agg["test/auroc_mean"] == pytest.approx(0.7)
    assert agg["test/auroc_n_runs"] == 2.0  # the NaN fold is counted out


def test_single_run_has_zero_spread_and_a_degenerate_interval():
    agg = aggregate_cv_metrics({"test/auroc": [0.75]}, [], log, 1, OFFSET)
    assert agg["test/auroc_mean"] == pytest.approx(0.75)
    assert agg["test/auroc_std"] == 0.0
    assert agg["test/auroc_ci_low"] == pytest.approx(0.75)
    assert agg["test/auroc_ci_high"] == pytest.approx(0.75)
    assert agg["test/auroc_n_runs"] == 1.0


def test_metric_with_no_finite_values_is_dropped_entirely():
    agg = aggregate_cv_metrics({"test/auroc": [float("nan"), float("nan")]}, [], log, 2, OFFSET)
    assert not any(k.startswith("test/auroc") for k in agg)


def test_summary_row_is_logged_once_after_the_per_run_rows():
    """The summary must sort after every per-run row so it cannot be mistaken
    for another fold."""
    rec = _RecordingLogger()
    aggregate_cv_metrics({"test/auroc": [0.8, 0.6]}, [rec], log, total_runs=2, log_step_offset=OFFSET)

    assert len(rec.calls) == 1
    logged, step = rec.calls[0]
    assert step == OFFSET + 2 + 1
    assert all(k.startswith("cv_summary/") for k in logged)


def test_log_step_offset_is_honoured():
    """The offset is a parameter here (cv_train.py hardcodes its constant), so
    a plan can supply its own."""
    rec = _RecordingLogger()
    aggregate_cv_metrics({"test/auroc": [0.8]}, [rec], log, total_runs=1, log_step_offset=500)
    _logged, step = rec.calls[0]
    assert step == 500 + 1 + 1

