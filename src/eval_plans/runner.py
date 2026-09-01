"""The one generic loop every evaluation plan runs through.

Nothing in this module knows what "walk-forward" or "grouped CV" means. It
executes whatever units a plan yields, records what each produced, and hands
the collection to the plan's ``aggregate()``. All strategy-specific behavior
lives in the plans; all orchestration lives here.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

import hydra
import lightning as L

from src.callbacks.prediction_collector import PredictionCollectorCallback
from src.eval_plans.base import CohortCache, EvalPlan, RunContext, RunUnit, UnitResult
from src.eval_plans.fold_identity import fold_identity
from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Each unit's trainer restarts its own step counter from zero, so a
# step-monotonic logger (W&B, TensorBoard) drops a per-unit row logged at a
# colliding step. Offsetting past any plausible real training step keeps the
# per-unit and summary rows distinct. Matches cv_train.py's CV_LOG_STEP_OFFSET
# and wf_cv_train.py's WF_LOG_STEP_OFFSET, both 1_000_000.
LOG_STEP_OFFSET = 1_000_000


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """Returns a copy of ``cfg`` with ``overrides`` applied. Never mutates ``cfg``.

    A deep copy per unit rather than in-place mutation (which is what the
    standalone CV scripts do under ``open_dict``) so nothing leaks between
    units: unit N's config cannot be perturbed by unit N-1's overrides, and the
    base config stays pristine for the aggregation step.

    ``force_add=True`` is required, not optional: Hydra composes configs with
    ``struct=True`` and ``deepcopy`` preserves that flag, so updating a key the
    data config does not already declare raises ``ConfigAttributeError``
    without it. The tradeoff is that a typo in an override key silently creates
    a new key instead of erroring, so every applied override is logged at debug
    level to make that diagnosable.
    """
    unit_cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        OmegaConf.update(unit_cfg, key, value, force_add=True)
        log.debug(f"override: {key}={value}")
    return unit_cfg


class CohortCache:
    """Builds at most one probe datamodule, and shares one cohort across units.

    Two entry points, mirroring how the standalone scripts already behave:

    - ``probe()`` -- a plan calls this when it needs a datamodule *before* the
      loop (to resolve a dynamic fold count). The probe's cohort is adopted, so
      the build is not wasted. Plans that never call it (single-split) trigger
      no build at all and add no database round-trip.
    - ``adopt(datamodule)`` -- the runner calls this after every unit's
      datamodule is ready; it is a no-op once a cohort is held. This reproduces
      cv_train.py's behavior of taking the cohort from fold 0 when no probe was
      needed.

    Building the cohort once and passing it to every subsequent datamodule as
    ``prebuilt_cohort`` is what keeps a multi-unit run from re-querying the
    database per unit.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._probe: Optional[LightningDataModule] = None
        self._cohort: Optional[Any] = None

    def probe(self) -> LightningDataModule:
        """Instantiates (once) a datamodule with the config's default fold, set
        up far enough that ``get_num_folds()``/``master_df`` are available.

        Uses the base config with no unit overrides, so ``current_fold`` sits at
        its configured default -- fold-list construction does not depend on
        which fold is selected, but ``_split_data`` does validate the selected
        one is in range, so a config defaulting ``current_fold`` out of range
        would fail here rather than mid-loop.
        """
        if self._probe is None:
            log.info("Building probe datamodule to resolve the plan's unit count...")
            self._probe = hydra.utils.instantiate(self._cfg.data)
            self.adopt(self._probe)
        return self._probe

    def adopt(self, datamodule: LightningDataModule) -> None:
        """Takes this datamodule's cohort if one is not already held."""
        if self._cohort is not None:
            return
        if not hasattr(datamodule, "get_prebuilt_cohort"):
            return
        if not hasattr(datamodule, "master_df"):
            datamodule.setup()
        self._cohort = datamodule.get_prebuilt_cohort()

    def shared_cohort(self) -> Optional[Any]:
        return self._cohort


def run_unit(
    unit_cfg: DictConfig,
    unit: RunUnit,
    plan: EvalPlan,
    loggers: List[Logger],
    cohort: CohortCache,
    log_hparams: bool,
) -> UnitResult:
    """Runs one unit end to end and returns what it produced.

    Instantiation order is datamodule -> model -> callbacks -> [collector] ->
    trainer, matching both standalone CV scripts exactly. This is not
    cosmetic: seeding happens once per *run*, not per unit, so every draw from
    the global RNG stream shifts what later units see. Reordering these calls,
    or adding an instantiation, changes every downstream unit's numbers.

    Each unit gets freshly instantiated objects for the same reason plus a
    second one: ``cfg.data`` instantiates a sampler and scaler recursively, and
    those are stateful (``LagSampler.set_labels`` mutates; ``SubjectScaler.fit``
    populates per-user state). Hoisting any of this out of the loop as an
    "optimization" would silently leak one unit's fitted state into the next.
    """
    datamodule: LightningDataModule = hydra.utils.instantiate(
        unit_cfg.data, prebuilt_cohort=cohort.shared_cohort()
    )
    cohort.adopt(datamodule)

    model: LightningModule = hydra.utils.instantiate(unit_cfg.model)
    callbacks: List[Callback] = instantiate_callbacks(unit_cfg.get("callbacks"))

    collector: Optional[PredictionCollectorCallback] = None
    trainer_callbacks = callbacks
    if plan.collect_predictions:
        collector = PredictionCollectorCallback(fold_index=unit.index)
        trainer_callbacks = callbacks + [collector]

    trainer: Trainer = hydra.utils.instantiate(
        unit_cfg.trainer, callbacks=trainer_callbacks, logger=loggers
    )

    if log_hparams and loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters({
            "cfg": unit_cfg, "datamodule": datamodule, "model": model,
            "callbacks": callbacks, "logger": loggers, "trainer": trainer,
        })

    if unit_cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=unit_cfg.get("ckpt_path"))

    if plan.requires_test or unit_cfg.get("test"):
        ckpt_path = (
            trainer.checkpoint_callback.best_model_path
            if getattr(trainer, "checkpoint_callback", None) else ""
        )
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    metrics = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in trainer.callback_metrics.items()
    }

    predictions = None
    if collector is not None:
        collected = collector.to_dataframe()
        predictions = collected[collected["stage"] == "test"].copy()

    return UnitResult(
        unit=unit,
        metrics=metrics,
        identity=fold_identity(datamodule),
        predictions=predictions,
        callbacks=callbacks,
    )


def run_plan(cfg: DictConfig, plan: EvalPlan) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Seeds, resolves the plan's units, runs each, then aggregates.

    Seeding happens exactly once here rather than per unit -- matching the
    standalone scripts, so their numbers are reproducible through this path.
    That does mean unit K's results depend on how much RNG units 0..K-1 drew;
    per-unit seeding would be more independently reproducible but would change
    every number relative to the existing scripts, so it is deliberately not
    done during the parity window.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    cohort = CohortCache(cfg)
    units = list(plan.units(cfg, cohort))
    if not units:
        raise ValueError(
            f"Eval plan '{plan.name}' produced zero units to run. Check the "
            "data config's fold parameters against the cohort."
        )
    log.info(f"Eval plan '{plan.name}': {len(units)} unit(s) to run.")

    results: List[UnitResult] = []
    cohort_hashes_seen: set = set()

    for unit in units:
        log.info(f"--- {unit.label} ({unit.index + 1}/{len(units)}) ---")
        unit_cfg = apply_overrides(cfg, unit.overrides)
        result = run_unit(
            unit_cfg, unit, plan, loggers, cohort, log_hparams=(unit.index == 0)
        )
        results.append(result)

        # ONE row per unit, merging the plan's tag, the cohort/fold fingerprint,
        # and this unit's metrics. It must be a single log_metrics call:
        # compare_cv_runs.py reads cv/repeat, cv/fold and fold/<metric> off the
        # same CSV row, and two calls would write two rows -- leaving it to
        # silently produce a table of NaNs rather than fail loudly.
        row: Dict[str, float] = {**unit.tag, **result.identity}
        if result.predictions is not None:
            row["cv/n_test_rows"] = float(len(result.predictions))
        row.update({f"fold/{k}": v for k, v in result.metrics.items()})

        for l in loggers:
            if hasattr(l, "log_metrics"):
                l.log_metrics(row, step=plan.step_for(unit))

        cohort_hash = row.get("cv/cohort_hash")
        if cohort_hash is not None:
            if cohort_hashes_seen and cohort_hash not in cohort_hashes_seen:
                log.warning(
                    "Cohort changed between units within this run — folds are "
                    "not drawn from a fixed set of users, so the aggregate below "
                    "mixes partitions of different cohorts."
                )
            cohort_hashes_seen.add(cohort_hash)

    ctx = RunContext(cfg=cfg, loggers=loggers, log=log, total_units=len(units))
    aggregated = plan.aggregate(results, ctx)

    # The last trainer has already torn down by this point, so nothing else will
    # flush the rows written above; without this they can be lost entirely for
    # buffered loggers such as CSVLogger.
    for l in loggers:
        if hasattr(l, "save"):
            l.save()

    object_dict = {
        "cfg": cfg,
        "logger": loggers,
        "plan": plan,
        "results": results,
        "per_run_metrics": [{**r.unit.tag, **r.identity} for r in results],
    }
    pooled = [r.predictions for r in results if r.predictions is not None]
    if pooled:
        object_dict["pooled_predictions"] = pd.concat(pooled, ignore_index=True)

    return aggregated, object_dict
