"""Unit tests for each EvalPlan's units()/aggregate(), with no Lightning,
no Trainer and no database -- a fake CohortCache stands in for the datamodule
probe, so these run in milliseconds.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.eval_plans import UserCVPlan, RunContext, SingleSplitPlan, UnitResult, WalkForwardPlan
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


# ----------------------------------------------------------------- single ---

def test_single_plan_yields_exactly_one_unit_with_no_overrides():
    cache = _FakeCohortCache()
    units = list(SingleSplitPlan().units(_cfg(), cache))

    assert len(units) == 1
    assert units[0].overrides == {}
    assert units[0].index == 0


def test_single_plan_never_probes():
    """The guarantee that eval_plan=single costs no extra cohort build."""
    cache = _FakeCohortCache()
    list(SingleSplitPlan().units(_cfg(), cache))
    assert cache.probe_calls == 0


def test_single_plan_tag_is_empty_so_compare_cv_runs_still_refuses_it():
    """compare_cv_runs.py identifies per-run rows by cv/repeat + cv/fold. A
    single split has neither, so omitting them keeps that tool correctly
    declining to pair single-split runs."""
    units = list(SingleSplitPlan().units(_cfg(), _FakeCohortCache()))
    assert units[0].tag == {}


def test_single_plan_aggregate_is_the_identity():
    plan = SingleSplitPlan()
    unit = list(plan.units(_cfg(), _FakeCohortCache()))[0]
    metrics = {"test/auroc": 0.81, "test/f1": 0.5}
    assert plan.aggregate([_result(unit, metrics)], _ctx()) == metrics


def test_single_plan_does_not_force_testing():
    assert SingleSplitPlan().requires_test is False
    assert SingleSplitPlan().collect_predictions is False


# ------------------------------------------------------------- grouped CV ---

def test_user_cv_yields_repeats_times_folds_units():
    cache = _FakeCohortCache()
    units = list(UserCVPlan().units(_cfg(num_folds=3, num_repeats=2), cache))

    assert len(units) == 6
    assert [(u.tag["cv/repeat"], u.tag["cv/fold"]) for u in units] == [
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
        (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    ]
    assert [u.tag["cv/run"] for u in units] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [u.index for u in units] == [0, 1, 2, 3, 4, 5]


def test_user_cv_overrides_select_the_fold_and_repeat():
    units = list(UserCVPlan().units(_cfg(num_folds=2, num_repeats=1), _FakeCohortCache()))
    assert units[1].overrides == {
        "data.current_fold": 1, "data.current_repeat": 0, "data.num_folds": 2,
    }


def test_user_cv_does_not_probe_for_a_fixed_fold_count():
    """Only leave-one-user-out needs the cohort up front; a fixed count must
    not pay for a probe build."""
    cache = _FakeCohortCache()
    list(UserCVPlan().units(_cfg(num_folds=5, num_repeats=1), cache))
    assert cache.probe_calls == 0


def test_user_cv_probes_once_to_resolve_leave_one_user_out():
    cache = _FakeCohortCache(_StubProbe(num_folds=7))
    units = list(UserCVPlan().units(_cfg(num_folds=-1, num_repeats=1), cache))

    assert cache.probe_calls == 1
    assert len(units) == 7
    # The resolved count replaces the -1 sentinel in every unit's config.
    assert all(u.overrides["data.num_folds"] == 7 for u in units)


def test_user_cv_raises_when_the_datamodule_cannot_count_folds():
    cache = _FakeCohortCache(_ProbeWithoutFoldCounting())
    with pytest.raises(ValueError, match="dynamic fold counting"):
        list(UserCVPlan().units(_cfg(num_folds=-1), cache))


def test_user_cv_defaults_to_five_folds_and_one_repeat():
    units = list(UserCVPlan().units(_cfg(), _FakeCohortCache()))
    assert len(units) == 5


def test_user_cv_aggregate_reports_mean_spread_and_interval():
    plan = UserCVPlan()
    units = list(plan.units(_cfg(num_folds=3, num_repeats=1), _FakeCohortCache()))
    results = [
        _result(units[0], {"test/auroc": 0.8}),
        _result(units[1], {"test/auroc": 0.6}),
        _result(units[2], {"test/auroc": 0.7}),
    ]
    agg = plan.aggregate(results, _ctx(total_units=3))

    assert agg["test/auroc_mean"] == pytest.approx(0.7)
    assert agg["test/auroc_std"] == pytest.approx(np.std([0.8, 0.6, 0.7], ddof=1))
    assert agg["test/auroc_n_runs"] == 3.0
    assert "cv_summary/test/auroc_mean" in agg


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


def test_user_cv_steps_are_one_based_within_the_offset():
    """Matches cv_train.py, which increments `run` before logging."""
    plan = UserCVPlan()
    units = list(plan.units(_cfg(num_folds=3, num_repeats=1), _FakeCohortCache()))
    assert [plan.step_for(u) for u in units] == [
        plan.log_step_offset + i + 1 for i in range(3)
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
