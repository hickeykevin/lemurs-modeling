"""Unit tests for each EvalPlan's units()/aggregate(), with no Lightning,
no Trainer and no database -- a fake CohortCache stands in for the datamodule
probe, so these run in milliseconds.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.eval_plans import RunContext, UnitResult, WalkForwardPlan
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class _StubProbe:
    def __init__(self, num_folds=4, fold_sizing=None):
        self._num_folds = num_folds
        # Mirrors the datamodule's Lightning hparams namespace, which the
        # walk-forward plan reads to name the right knobs in its zero-fold
        # error. fold_sizing=None stands for a probe with no hparams at all.
        if fold_sizing is not None:
            self.hparams = SimpleNamespace(fold_sizing=fold_sizing)

    def get_num_folds(self):
        return self._num_folds


class _ProbeWithoutFoldCounting:
    """Mimics a datamodule that has no get_num_folds() (e.g. plain HealthDataModule)."""


class _FakeCohortCache:
    """Records whether a plan actually asked for a probe.

    The probe-call count is the assertion that protects the "single-split adds
    no database round-trip" guarantee -- a regression there would otherwise
    only surface as an unexplained slowdown on real runs.
    """

    def __init__(self, probe=None):
        self._probe = probe if probe is not None else _StubProbe()
        self.probe_calls = 0

    def probe(self):
        self.probe_calls += 1
        return self._probe

    def shared_cohort(self):
        return None


def _cfg(**data):
    return OmegaConf.create({"data": dict(data), "seed": 7})


def _ctx(cfg=None, total_units=1, loggers=None):
    return RunContext(
        cfg=cfg if cfg is not None else _cfg(),
        loggers=loggers or [],
        log=log,
        total_units=total_units,
    )


def _result(unit, metrics, predictions=None, callbacks=None):
    return UnitResult(
        unit=unit, metrics=metrics, identity={},
        predictions=predictions, callbacks=callbacks or [],
    )


# ----------------------------------------------------------- walk-forward ---

def test_walk_forward_yields_one_unit_per_fold_with_no_repeat_axis():
    cache = _FakeCohortCache(_StubProbe(num_folds=5))
    units = list(WalkForwardPlan().units(_cfg(), cache))

    assert len(units) == 5
    assert [u.overrides for u in units] == [{"data.current_fold": i} for i in range(5)]
    # No repeat axis: a walk-forward fold sequence is not an independently
    # reshuffled draw, so re-running under a new seed answers no new question.
    assert all("cv/repeat" not in u.tag for u in units)
    assert [u.tag["cv/fold"] for u in units] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_walk_forward_always_probes_because_fold_count_is_cohort_dependent():
    cache = _FakeCohortCache(_StubProbe(num_folds=3))
    list(WalkForwardPlan().units(_cfg(), cache))
    assert cache.probe_calls == 1


def test_walk_forward_raises_on_a_datamodule_without_fold_counting():
    cache = _FakeCohortCache(_ProbeWithoutFoldCounting())
    with pytest.raises(ValueError, match="get_num_folds"):
        list(WalkForwardPlan().units(_cfg(), cache))


def test_walk_forward_raises_when_the_cohort_supports_zero_folds():
    cache = _FakeCohortCache(_StubProbe(num_folds=0))
    with pytest.raises(ValueError, match="zero walk-forward folds"):
        list(WalkForwardPlan().units(_cfg(), cache))


@pytest.mark.parametrize(
    "fold_sizing, expected_knobs",
    [
        ("cyclic", "train_width_pct/step_pct"),
        ("pct", "burn_in_pct/val_pct/step_pct"),
        # No hparams at all -> a generic phrase, since there is no third mode
        # whose knobs we could name.
        (None, "this config's fold-sizing parameters"),
    ],
)
def test_walk_forward_zero_fold_error_names_this_configs_own_knobs(
    fold_sizing, expected_knobs
):
    """The remedy must name keys the running config actually has.

    This used to name a removed count mode's parameters unconditionally,
    pointing at keys no config in this repo defines.
    """
    cache = _FakeCohortCache(_StubProbe(num_folds=0, fold_sizing=fold_sizing))
    with pytest.raises(ValueError) as excinfo:
        list(WalkForwardPlan().units(_cfg(), cache))

    message = str(excinfo.value)
    assert expected_knobs in message
    for other in ("train_width_pct/step_pct", "burn_in_pct/val_pct/step_pct",
                  "this config's fold-sizing parameters"):
        if other != expected_knobs:
            assert other not in message


def test_walk_forward_requires_testing_and_prediction_collection():
    """Its aggregation IS the pooling, so honouring test=False would leave
    nothing to pool."""
    plan = WalkForwardPlan()
    assert plan.requires_test is True
    assert plan.collect_predictions is True


def test_walk_forward_steps_are_zero_based_within_the_offset():
    """Matches wf_cv_train.py's existing rows, so a diff against it is clean."""
    plan = WalkForwardPlan()
    units = list(plan.units(_cfg(), _FakeCohortCache(_StubProbe(num_folds=3))))
    assert [plan.step_for(u) for u in units] == [
        plan.log_step_offset + i for i in range(3)
    ]


def _pred_frame(fold_index, n=6, seed=0):
    rng = np.random.RandomState(seed)
    score = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "fold_index": fold_index,
        "stage": "test",
        "app_user_id": [i % 3 for i in range(n)],
        "record_timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
        "y_true": [i % 2 for i in range(n)],
        "prob_class_0": 1 - score,
        "prob_class_1": score,
    })


def test_walk_forward_aggregate_pools_every_folds_predictions(tmp_path):
    """Pooling, not fold-averaging: a fold index here is a per-user position,
    not a shared calendar window, so per-fold scores are not comparable."""
    plan = WalkForwardPlan(n_bootstraps=20)  # structure test, not CI precision
    units = list(plan.units(_cfg(), _FakeCohortCache(_StubProbe(num_folds=2))))
    results = [
        _result(units[0], {}, predictions=_pred_frame(0, seed=1)),
        _result(units[1], {}, predictions=_pred_frame(1, seed=2)),
    ]
    cfg = OmegaConf.create({"data": {}, "paths": {"output_dir": str(tmp_path)}})
    metrics = plan.aggregate(results, _ctx(cfg=cfg, total_units=2))

    assert metrics["pooled/n_predictions"] == 12.0  # both folds, pooled

    saved = pd.read_csv(tmp_path / "pooled_predictions.csv")
    assert len(saved) == 12
    assert sorted(saved["fold_index"].unique()) == [0, 1]


def test_walk_forward_aggregate_skips_saving_when_there_is_no_output_dir():
    plan = WalkForwardPlan(n_bootstraps=20)
    units = list(plan.units(_cfg(), _FakeCohortCache(_StubProbe(num_folds=1))))
    results = [_result(units[0], {}, predictions=_pred_frame(0, seed=3))]
    metrics = plan.aggregate(results, _ctx(cfg=OmegaConf.create({"data": {}}), total_units=1))
    assert "pooled/n_predictions" in metrics  # metrics still computed
