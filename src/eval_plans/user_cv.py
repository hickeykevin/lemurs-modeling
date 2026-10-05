from collections import defaultdict
from typing import Dict, Iterator, List

import pandas as pd
from omegaconf import DictConfig

from src.eval_plans.base import (
    CohortCache,
    RunContext,
    RunUnit,
    UnitResult,
    default_step_for,
)
from src.eval_plans.cv_aggregation import aggregate_cv_metrics
from src.eval_plans.runner import LOG_STEP_OFFSET
from src.eval_plans.walk_forward import WalkForwardPlan



class UserCVPlan:
    """Repeated, user-grouped k-fold -- "does this generalize to a NEW user?".

    Every split is grouped on ``app_user_id`` so no participant appears on both
    sides, which is the arrangement that matches deployment: a new user of the
    app has taken no surveys, so the model has seen none of their labels. This
    is a different question from the walk-forward plan's ("given a user's own
    history, can we forecast their near-future risk?") and the two are not
    interchangeable.

    Results combine as mean +/- sample sd with a 95% interval, not by pooling
    predictions: each fold's test users are a genuine independent draw from the
    cohort, so the spread across folds *is* the quantity of interest. At this
    cohort size which users land in a fold moves the result more than most
    modelling choices do, which is why the spread is reported as a first-class
    output rather than a diagnostic.

    ``num_folds: -1`` means leave-one-user-out, resolved by probing the
    datamodule for its fold count. That is the only case where this plan builds
    a probe -- a fixed fold count needs none, and the cohort is then adopted
    from unit 0 instead.
    """

    name = "user_cv"
    log_step_offset = LOG_STEP_OFFSET
    requires_test = False  # honours cfg.test, matching cv_train.py
    # Per-survey test predictions are saved (not used for the headline metrics,
    # which stay per-fold) so held-out users' predictions can be inspected, e.g.
    # within-person AUROC on people the model never trained on.
    collect_predictions = True

    def units(self, cfg: DictConfig, cohort: CohortCache) -> Iterator[RunUnit]:
        num_folds = cfg.data.get("num_folds", 5)
        if num_folds == -1:
            probe = cohort.probe()
            if not hasattr(probe, "get_num_folds"):
                raise ValueError(
                    "The configured datamodule does not support dynamic fold counting (-1)."
                )
            num_folds = probe.get_num_folds()
            # Write the resolved count back so the per-unit config (and the
            # hyperparameters logged from it) record the real number rather
            # than the -1 sentinel.

        num_repeats = cfg.data.get("num_repeats", 1)
        total = num_repeats * num_folds

        run = 0
        for repeat in range(num_repeats):
            for fold in range(num_folds):
                run += 1
                yield RunUnit(
                    overrides={
                        "data.current_fold": fold,
                        "data.current_repeat": repeat,
                        "data.num_folds": num_folds,
                    },
                    # compare_cv_runs.py reads cv/repeat and cv/fold to identify
                    # per-run rows and pair two sweeps fold by fold; cv/run is
                    # the flat index. This key set is a downstream contract.
                    tag={
                        "cv/repeat": float(repeat),
                        "cv/fold": float(fold),
                        "cv/run": float(run),
                    },
                    index=run - 1,
                    label=(
                        f"repeat {repeat + 1}/{num_repeats}, "
                        f"fold {fold + 1}/{num_folds} (run {run}/{total})"
                    ),
                )

    def aggregate(self, results: List[UnitResult], ctx: RunContext) -> Dict[str, float]:
        metrics_by_run: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for key, value in result.metrics.items():
                metrics_by_run[key].append(value)
        self._save_predictions(results, ctx)
        return aggregate_cv_metrics(
            metrics_by_run, ctx.loggers, ctx.log, ctx.total_units, self.log_step_offset
        )

    @staticmethod
    def _save_predictions(results: List[UnitResult], ctx: RunContext) -> None:
        """Writes every unit's test predictions to ``pooled_predictions.csv``.

        Each row is tagged with its ``repeat`` and ``cv_fold``; ``fold_index`` is
        the flat unit index. Within one repeat every user is tested exactly once,
        so a single repeat's rows form one complete out-of-fold prediction set.
        """
        frames = []
        for result in results:
            if result.predictions is None or result.predictions.empty:
                continue
            frames.append(result.predictions.assign(
                repeat=int(result.unit.tag["cv/repeat"]),
                cv_fold=int(result.unit.tag["cv/fold"]),
            ))
        if frames:
            WalkForwardPlan._save_pooled_predictions(ctx, pd.concat(frames, ignore_index=True))

    def step_for(self, unit: RunUnit) -> int:
        return default_step_for(self, unit)
