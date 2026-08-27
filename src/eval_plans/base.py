"""Core shapes for the evaluation-plan architecture.

An ``EvalPlan`` describes an evaluation *strategy* as data rather than as
orchestration code: which units of work exist (``units()``) and how their
results combine into one reported number (``aggregate()``). One generic runner
(``src.eval_plans.runner.run_plan``) executes every plan identically --
instantiate datamodule/model/callbacks/trainer, fit, test, record -- so a new
strategy costs a small ``units()``/``aggregate()`` pair, not a fourth
orchestration script.

The strategies this replaces differ along only a few axes:

===================  ==========  =====================  ====================
axis                 single      grouped CV             walk-forward
===================  ==========  =====================  ====================
varies per unit      nothing     current_fold+repeat    current_fold
unit count           1           repeats x folds        get_num_folds()
combine results      identity    mean +/- sd, 95% CI    pool, then metrics
testing              cfg.test    cfg.test               always (needs preds)
===================  ==========  =====================  ====================

Note that "cyclic" is deliberately absent: it is a ``fold_sizing`` value on
``WalkForwardHealthDataModule``, not a distinct loop shape, so it needs no plan
of its own -- ``eval_plan=walk_forward data=walk_forward_cyclic_5fold_sweep``
already expresses it. A plan is warranted only when ``units()`` or
``aggregate()`` genuinely differs.

KNOWN BOUNDARY OF THIS ABSTRACTION
----------------------------------
``units()`` is fixed before the first unit runs. A strategy whose later units
depend on earlier *results* -- nested CV (an inner loop selecting
hyperparameters for an outer loop's evaluation), or adaptive early stopping
across folds -- cannot be expressed here and needs a different protocol. That
boundary is stated rather than speculatively engineered around: ``units()``
returns an ``Iterator`` so a future protocol could feed results back via
``send()``, but no such machinery is built today.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import pandas as pd
from lightning import Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import RankedLogger


@dataclass(frozen=True)
class RunUnit:
    """One (fit, test) execution the runner will perform.

    Args:
        overrides: Dotted config paths to values, applied to a *copy* of the
            base config for this unit (see ``runner.apply_overrides``). This is
            how a plan varies ``data.current_fold`` and friends without
            mutating shared state between units.
        tag: Scalar columns merged into this unit's single logged row. Keys
            here are part of a real downstream contract -- ``compare_cv_runs.py``
            requires ``cv/repeat`` and ``cv/fold`` to identify a per-run row --
            so a plan chooses them deliberately (see each plan's docstring).
        index: 0-based position in the plan's sequence.
        label: Human-readable progress label, e.g. "repeat 1/10, fold 3/5".
    """

    overrides: Dict[str, Any]
    tag: Dict[str, float]
    index: int
    label: str


@dataclass
class UnitResult:
    """What one unit produced, handed to ``EvalPlan.aggregate``."""

    unit: RunUnit
    metrics: Dict[str, float]
    identity: Dict[str, float]
    predictions: Optional[pd.DataFrame] = None
    callbacks: List[Callback] = field(default_factory=list)


@dataclass
class RunContext:
    """Everything ``aggregate()`` may legitimately need beyond the results.

    Passed as one object rather than several positionals because different
    plans need different subsets: walk-forward needs ``loggers`` (to log pooled
    metrics at its own step) and ``cfg`` (for ``paths.output_dir``), while
    grouped CV needs only ``loggers`` and ``total_units``.
    """

    cfg: DictConfig
    loggers: List[Logger]
    log: RankedLogger
    total_units: int


@runtime_checkable
class CohortCache(Protocol):
    """Lazily builds at most one datamodule for probing, and shares its cohort.

    A plan calls ``probe()`` only if it needs a datamodule *before* the loop --
    to resolve a dynamic fold count. Plans that don't (e.g. single-split) never
    trigger a build, so no database round-trip is added for them.
    """

    def probe(self) -> LightningDataModule: ...

    def shared_cohort(self) -> Optional[Any]: ...


@runtime_checkable
class EvalPlan(Protocol):
    """An evaluation strategy, expressed as units of work plus an aggregation.

    Attributes:
        name: Short identifier used in logs.
        log_step_offset: Base step for this plan's logged rows. Each fold's
            trainer restarts its own step counter at zero, so a step-monotonic
            logger (W&B, TensorBoard) would drop per-unit rows logged at
            colliding steps without an offset well past any real training step.
        requires_test: When True the runner tests unconditionally, ignoring
            ``cfg.test`` -- for plans whose aggregation needs predictions, where
            skipping the test pass would leave nothing to aggregate.
        collect_predictions: When True the runner attaches a
            ``PredictionCollectorCallback`` per unit and puts the collected
            test-stage frame on ``UnitResult.predictions``.
    """

    name: str
    log_step_offset: int
    requires_test: bool
    collect_predictions: bool

    def units(self, cfg: DictConfig, cohort: CohortCache) -> Iterator[RunUnit]: ...

    def aggregate(self, results: List[UnitResult], ctx: RunContext) -> Dict[str, float]: ...

    def step_for(self, unit: RunUnit) -> int: ...


def default_step_for(plan: "EvalPlan", unit: RunUnit) -> int:
    """Default logged step for a unit: 1-based within the plan's offset.

    Plans that need a different convention override ``step_for`` -- the
    walk-forward plan uses a 0-based step to match ``wf_cv_train.py``'s
    existing output exactly.
    """
    return plan.log_step_offset + unit.index + 1
