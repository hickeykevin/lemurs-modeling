from typing import Dict, Iterator, List

from omegaconf import DictConfig

from src.eval_plans.base import (
    CohortCache,
    RunContext,
    RunUnit,
    UnitResult,
    default_step_for,
)


class SingleSplitPlan:
    """One train/val/test split, evaluated once -- the baseline strategy.

    Reproduces what ``train.py`` did before the eval_plan layer existed: a
    single fit/test with no fold loop and no cross-unit aggregation. Whether a
    test pass runs is left to ``cfg.test`` (``requires_test=False``), and
    ``aggregate`` is the identity on the one unit's metrics.

    ``units()`` never calls ``cohort.probe()``, so no probe datamodule is built
    and this plan adds no database round-trip over the pre-eval_plan behavior.
    ``tests/test_eval_plans.py`` asserts that directly, since a regression there
    would only otherwise show up as an unexplained slowdown.

    The unit's ``tag`` is deliberately empty: ``compare_cv_runs.py`` identifies
    per-run rows by the presence of ``cv/repeat``/``cv/fold``, and a single
    split has neither, so omitting them keeps that tool correctly refusing to
    pair single-split runs rather than silently producing a one-row "CV" table.

    Pooled, user-cluster bootstrap CIs are available to this plan the same way
    they always were -- via ``callbacks=pooled_eval``, which attaches
    ``PooledMetricsCallback`` (see ``docs/eval_schemes.md``). That is a
    per-trainer concern, unrelated to ``collect_predictions``, which exists for
    plans that pool *across* units.
    """

    name = "single"
    log_step_offset = 0
    requires_test = False
    collect_predictions = False

    def units(self, cfg: DictConfig, cohort: CohortCache) -> Iterator[RunUnit]:
        yield RunUnit(overrides={}, tag={}, index=0, label="single split")

    def aggregate(self, results: List[UnitResult], ctx: RunContext) -> Dict[str, float]:
        return dict(results[0].metrics)

    def step_for(self, unit: RunUnit) -> int:
        return default_step_for(self, unit)
