import os
from typing import Dict, Iterator, List

import pandas as pd
from omegaconf import DictConfig

from src.eval_plans.base import CohortCache, RunContext, RunUnit, UnitResult
from src.eval_plans.runner import LOG_STEP_OFFSET
from src.eval_plans.pooled_metrics import compute_pooled_metrics, fold_breakdown_table



class WalkForwardPlan:
    """Per-user walk-forward CV -- "given a user's own history, can the model
    forecast their near-future risk?".

    Covers every ``WalkForwardHealthDataModule`` fold-construction mode, since
    they differ only in how ``CohortSplitter`` cuts folds, not in how the folds
    are run or combined. In particular **cyclic needs no plan class of its own**:
    ``eval_plan=cyclical`` is a config that instantiates *this* class and swaps
    the data config to the cyclic sweep, and ``eval_plan=walk_forward`` is the
    expanding-window one. See ``configs/eval_plan/cyclical.yaml`` and
    ``docs/eval_schemes.md``.

    Results are combined by pooling every fold's out-of-fold test predictions
    into one evaluation set and computing metrics once, rather than averaging
    per-fold scores. This is not a stylistic choice: a fold index here is a
    per-user *position* in that user's own timeline, not a shared calendar
    window across users, so fold 2 for one participant and fold 2 for another
    are not measuring a comparable thing and their scores should not be averaged
    as if they were. A per-fold breakdown is still reported as a diagnostic (to
    see whether forecasting improves as history accumulates), but it is not the
    headline number.

    Because the aggregation *is* the pooling, testing is mandatory
    (``requires_test``): honouring a ``test=False`` here would leave nothing to
    pool. Prediction collection is likewise required
    (``collect_predictions``) -- the runner attaches a
    ``PredictionCollectorCallback`` per unit, tagged with the fold index, so
    pooled rows can be traced back to (participant, timestamp).

    There is no repeat axis, unlike ``UserCVPlan``. A walk-forward fold
    sequence is not an independently-reshuffled draw the way a grouped k-fold's
    assignment is, so re-running it under a different seed would not answer a
    new question.

    Note the ``tag`` carries ``cv/fold`` but no ``cv/repeat``, so
    ``compare_cv_runs.py`` will decline to pair walk-forward runs. That is
    correct rather than accidental: its pairing model is a repeat x fold grid
    over a shared cohort partition, which walk-forward folds are not.
    """

    name = "walk_forward"
    log_step_offset = LOG_STEP_OFFSET
    requires_test = True
    collect_predictions = True

    def __init__(self, n_bootstraps: int = 1000) -> None:
        """Args:
            n_bootstraps: BCa bootstrap resamples per pooled metric. The default
                is production quality; lower it only for tests that check
                structure rather than CI precision, since 8 metrics x
                n_bootstraps resamples dominates this plan's aggregation cost.
        """
        self.n_bootstraps = n_bootstraps

    def units(self, cfg: DictConfig, cohort: CohortCache) -> Iterator[RunUnit]:
        # The fold count depends on the cohort's actual per-user response-count
        # distribution, so it can only be resolved by building the cohort. The
        # probe's cohort is adopted by the cache, so every unit below reuses it
        # rather than re-querying the database.
        probe = cohort.probe()
        if not hasattr(probe, "get_num_folds"):
            raise ValueError(
                "cfg.data must instantiate a datamodule with get_num_folds() "
                "(e.g. WalkForwardHealthDataModule) for the walk_forward eval plan."
            )
        num_folds = probe.get_num_folds()
        if num_folds == 0:
            # Which knobs to lower depends on fold_sizing, so name the ones
            # this config actually has -- see configs/data/walk_forward_*.yaml.
            sizing = getattr(getattr(probe, "hparams", None), "fold_sizing", None)
            knobs = {
                "pct": "burn_in_pct/val_pct/step_pct",
                "cyclic": "train_width_pct/step_pct",
            }.get(sizing, "this config's fold-sizing parameters")
            raise ValueError(
                f"This cohort/config produces zero walk-forward folds (fold_sizing="
                f"{sizing!r}) -- lower {knobs}, or check the cohort's "
                "response-count distribution."
            )

        for fold in range(num_folds):
            yield RunUnit(
                overrides={"data.current_fold": fold},
                tag={"cv/fold": float(fold)},
                index=fold,
                label=f"fold {fold + 1}/{num_folds}",
            )

    def step_for(self, unit: RunUnit) -> int:
        # 0-based within the offset, matching wf_cv_train.py's existing rows.
        return self.log_step_offset + unit.index

    def aggregate(self, results: List[UnitResult], ctx: RunContext) -> Dict[str, float]:
        frames = [r.predictions for r in results if r.predictions is not None]
        pooled_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Reuse the run's own ClassificationMetricsCallback settings so the
        # pooled metrics use identical averaging/threshold params rather than a
        # second, possibly-drifted set of defaults. Every unit instantiates
        # callbacks from the same config, so the last unit's instance is
        # representative.
        classification_metrics_callback = None
        if results:
            classification_metrics_callback = next(
                (cb for cb in results[-1].callbacks
                 if type(cb).__name__ == "ClassificationMetricsCallback"),
                None,
            )

        pooled_metrics = compute_pooled_metrics(
            pooled_df, ctx.log,
            classification_metrics_callback=classification_metrics_callback,
            n_bootstraps=self.n_bootstraps,
        )
        fold_breakdown_table(pooled_df, ctx.log)

        for l in ctx.loggers:
            if hasattr(l, "log_metrics"):
                l.log_metrics(
                    pooled_metrics,
                    step=self.log_step_offset + ctx.total_units + 1,
                )

        self._save_pooled_predictions(ctx, pooled_df)
        return pooled_metrics

    @staticmethod
    def _save_pooled_predictions(ctx: RunContext, pooled_df: pd.DataFrame) -> None:
        """Persists the full pooled prediction table alongside the run's logs.

        Not just the summary metrics: this table is what a reported pooled
        AUROC/CI is computed from, and it should stay inspectable and
        recomputable afterwards.
        """
        cfg = ctx.cfg
        output_dir = (
            cfg.paths.output_dir
            if "paths" in cfg and "output_dir" in cfg.paths else None
        )
        if output_dir is None or pooled_df.empty:
            return
        out_path = os.path.join(output_dir, "pooled_predictions.csv")
        os.makedirs(output_dir, exist_ok=True)
        pooled_df.to_csv(out_path, index=False)
        ctx.log.info(f"Saved pooled predictions ({len(pooled_df)} rows) to {out_path}")
