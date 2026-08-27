"""Unit tests for the generic runner: config override isolation, the cohort
cache's build-once semantics, and the one-row-per-unit logging contract that
compare_cv_runs.py depends on.

run_plan is exercised with run_unit monkeypatched to return canned results, so
these tests touch no Lightning, no model and no database -- the loop's own
behavior is what's under test, not training.
"""

from pathlib import Path

import pandas as pd
import pytest
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

import src.eval_plans.runner as runner_mod
from src.eval_plans import GroupedCVPlan, RunUnit, UnitResult, apply_overrides, run_plan
from src.eval_plans.runner import CohortCache


# ------------------------------------------------------- apply_overrides ---

def _struct_cfg():
    """A config in the same shape Hydra produces: struct-mode enabled."""
    cfg = OmegaConf.create({"data": {"current_fold": 0}, "seed": 7, "trainer": {}})
    OmegaConf.set_struct(cfg, True)
    return cfg


def test_apply_overrides_does_not_mutate_the_base_config():
    """Per-unit isolation: unit N must not inherit unit N-1's overrides, and
    the base config must stay pristine for the aggregation step."""
    cfg = _struct_cfg()
    unit_cfg = apply_overrides(cfg, {"data.current_fold": 3})

    assert unit_cfg.data.current_fold == 3
    assert cfg.data.current_fold == 0


def test_apply_overrides_can_add_keys_absent_from_the_config():
    """force_add is required, not optional: Hydra composes with struct=True, so
    setting a key the data config doesn't declare would otherwise raise. This
    is exactly the case a future eval plan hits."""
    cfg = _struct_cfg()
    unit_cfg = apply_overrides(cfg, {"data.a_key_no_config_declares": 42})
    assert unit_cfg.data.a_key_no_config_declares == 42


def test_apply_overrides_leaves_interpolations_following_the_new_value():
    """A plan that varies `seed` must have it propagate to data.random_state,
    which every sweep data config interpolates from it."""
    cfg = OmegaConf.create({"seed": 7, "data": {"random_state": "${seed}"}})
    OmegaConf.set_struct(cfg, True)
    unit_cfg = apply_overrides(cfg, {"seed": 99})
    assert unit_cfg.data.random_state == 99


def test_apply_overrides_with_no_overrides_is_an_equal_copy():
    cfg = _struct_cfg()
    unit_cfg = apply_overrides(cfg, {})
    assert unit_cfg == cfg
    assert unit_cfg is not cfg


# ----------------------------------------------------------- CohortCache ---

class _StubDataModule:
    def __init__(self, cohort="COHORT", with_master=True):
        self._cohort = cohort
        self.setup_calls = 0
        if with_master:
            self.master_df = pd.DataFrame({"app_user_id": [1, 2]})

    def setup(self, stage=None):
        self.setup_calls += 1
        self.master_df = pd.DataFrame({"app_user_id": [1, 2]})

    def get_prebuilt_cohort(self):
        return self._cohort


class _NoCohortDataModule:
    """A datamodule with no cohort-sharing support (e.g. a plain one)."""


def test_cohort_cache_starts_empty():
    assert CohortCache(OmegaConf.create({"data": {}})).shared_cohort() is None


def test_cohort_cache_adopts_the_first_datamodule_only():
    cache = CohortCache(OmegaConf.create({"data": {}}))
    cache.adopt(_StubDataModule(cohort="FIRST"))
    cache.adopt(_StubDataModule(cohort="SECOND"))
    assert cache.shared_cohort() == "FIRST"


def test_cohort_cache_sets_up_a_datamodule_that_has_not_built_its_cohort():
    cache = CohortCache(OmegaConf.create({"data": {}}))
    dm = _StubDataModule(with_master=False)
    cache.adopt(dm)
    assert dm.setup_calls == 1
    assert cache.shared_cohort() == "COHORT"


def test_cohort_cache_does_not_set_up_an_already_built_datamodule():
    cache = CohortCache(OmegaConf.create({"data": {}}))
    dm = _StubDataModule(with_master=True)
    cache.adopt(dm)
    assert dm.setup_calls == 0


def test_cohort_cache_ignores_datamodules_without_cohort_support():
    cache = CohortCache(OmegaConf.create({"data": {}}))
    cache.adopt(_NoCohortDataModule())
    assert cache.shared_cohort() is None


# ------------------------------------------------- run_plan logging contract ---

class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((dict(metrics), step))

    def save(self):
        pass


def _canned_run_unit(metrics_by_index, identity=None):
    """Builds a run_unit stand-in that returns fixed metrics per unit index."""
    def _fake(unit_cfg, unit, plan, loggers, cohort, log_hparams):
        return UnitResult(
            unit=unit,
            metrics=metrics_by_index[unit.index],
            identity=dict(identity or {"cv/cohort_hash": 123.0}),
        )
    return _fake


def _patch(monkeypatch, fake):
    monkeypatch.setattr(runner_mod, "run_unit", fake)
    monkeypatch.setattr(runner_mod, "instantiate_loggers", lambda _cfg: [])


def test_run_plan_emits_exactly_one_row_per_unit(monkeypatch):
    """The contract compare_cv_runs.py depends on: a unit's tag, fingerprint
    and metrics must land on ONE row. Two log_metrics calls would write two
    CSV rows and leave that tool producing a table of NaNs."""
    rec = _RecordingLogger()
    metrics = {0: {"test/auroc": 0.8}, 1: {"test/auroc": 0.6}}
    _patch(monkeypatch, _canned_run_unit(metrics))
    monkeypatch.setattr(runner_mod, "instantiate_loggers", lambda _cfg: [rec])

    cfg = OmegaConf.create({"data": {"num_folds": 2, "num_repeats": 1}, "seed": None})
    run_plan(cfg, GroupedCVPlan())

    per_unit = [c for c in rec.calls if "cv/fold" in c[0]]
    assert len(per_unit) == 2
    row, _step = per_unit[0]
    # tag + identity + metrics, all together
    assert row["cv/repeat"] == 0.0 and row["cv/fold"] == 0.0 and row["cv/run"] == 1.0
    assert row["cv/cohort_hash"] == 123.0
    assert row["fold/test/auroc"] == 0.8


def test_run_plan_rows_are_parseable_by_compare_cv_runs(tmp_path, monkeypatch):
    """End-to-end through a real CSVLogger: the rows this runner writes must
    survive a round trip and satisfy compare_cv_runs.load_per_run_rows."""
    from src.compare_cv_runs import load_per_run_rows

    metrics = {i: {"test/auroc": 0.5 + i / 100} for i in range(4)}
    _patch(monkeypatch, _canned_run_unit(metrics))
    csv_logger = CSVLogger(save_dir=str(tmp_path), name="", version="")
    monkeypatch.setattr(runner_mod, "instantiate_loggers", lambda _cfg: [csv_logger])

    cfg = OmegaConf.create({"data": {"num_folds": 2, "num_repeats": 2}, "seed": None})
    run_plan(cfg, GroupedCVPlan())

    rows = load_per_run_rows(Path(csv_logger.log_dir))
    assert len(rows) == 4
    # cv/fold and the metric must be non-null on the SAME rows.
    assert rows["fold/test/auroc"].notna().all()
    assert sorted(rows.index.tolist()) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_run_plan_logs_hyperparameters_only_for_the_first_unit(monkeypatch):
    seen = []

    def _fake(unit_cfg, unit, plan, loggers, cohort, log_hparams):
        seen.append((unit.index, log_hparams))
        return UnitResult(unit=unit, metrics={"test/auroc": 0.5}, identity={})

    _patch(monkeypatch, _fake)
    cfg = OmegaConf.create({"data": {"num_folds": 3, "num_repeats": 1}, "seed": None})
    run_plan(cfg, GroupedCVPlan())

    assert seen == [(0, True), (1, False), (2, False)]


def test_run_plan_passes_each_units_overrides_through_to_run_unit(monkeypatch):
    seen_folds = []

    def _fake(unit_cfg, unit, plan, loggers, cohort, log_hparams):
        seen_folds.append(unit_cfg.data.current_fold)
        return UnitResult(unit=unit, metrics={"test/auroc": 0.5}, identity={})

    _patch(monkeypatch, _fake)
    cfg = OmegaConf.create({"data": {"num_folds": 3, "num_repeats": 1, "current_fold": 0}, "seed": None})
    OmegaConf.set_struct(cfg, True)
    run_plan(cfg, GroupedCVPlan())

    assert seen_folds == [0, 1, 2]


def test_run_plan_warns_when_the_cohort_changes_between_units(monkeypatch, caplog):
    """A shifting cohort means folds aren't drawn from a fixed set of users, so
    the aggregate would mix partitions of different cohorts."""
    calls = {"n": 0}

    def _fake(unit_cfg, unit, plan, loggers, cohort, log_hparams):
        calls["n"] += 1
        return UnitResult(
            unit=unit, metrics={"test/auroc": 0.5},
            identity={"cv/cohort_hash": float(calls["n"])},  # different every unit
        )

    _patch(monkeypatch, _fake)
    cfg = OmegaConf.create({"data": {"num_folds": 2, "num_repeats": 1}, "seed": None})
    with caplog.at_level("WARNING"):
        run_plan(cfg, GroupedCVPlan())

    assert any("Cohort changed between units" in r.message for r in caplog.records)


def test_run_plan_raises_when_a_plan_produces_no_units(monkeypatch):
    class _EmptyPlan:
        name = "empty"
        log_step_offset = 0
        requires_test = False
        collect_predictions = False

        def units(self, cfg, cohort):
            return iter([])

        def aggregate(self, results, ctx):
            return {}

        def step_for(self, unit):
            return 0

    _patch(monkeypatch, _canned_run_unit({}))
    with pytest.raises(ValueError, match="zero units"):
        run_plan(OmegaConf.create({"data": {}, "seed": None}), _EmptyPlan())


def test_run_plan_returns_the_plans_aggregate_and_the_unit_results(monkeypatch):
    metrics = {0: {"test/auroc": 0.8}, 1: {"test/auroc": 0.6}}
    _patch(monkeypatch, _canned_run_unit(metrics))

    cfg = OmegaConf.create({"data": {"num_folds": 2, "num_repeats": 1}, "seed": None})
    agg, obj = run_plan(cfg, GroupedCVPlan())

    assert agg["test/auroc_mean"] == pytest.approx(0.7)
    assert len(obj["results"]) == 2
    assert obj["plan"].name == "grouped_cv"
