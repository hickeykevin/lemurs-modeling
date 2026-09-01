from typing import Optional

import pandas as pd
from lightning import Callback, LightningModule, Trainer

from src.callbacks.prediction_collector import PredictionCollectorCallback
from src.eval_plans.pooled_metrics import compute_pooled_metrics
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)



class PooledMetricsCallback(Callback):
    """Attach this to get pooled, user-cluster BCa bootstrap CI metrics for a
    single ``trainer.test()`` run, instead of
    ``ClassificationMetricsCallback``'s row-level ``torchmetrics.BootStrapper``
    CI, which understates uncertainty here for the same reason a row-level
    bootstrap is wrong everywhere else in this pipeline: the outcome label
    is a person-level trait, and positives are concentrated in a handful of
    participants, so resampling individual responses (as if they were
    independent draws) understates how much the score would move under a
    different draw of *people* -- see ``CohortSplitter``'s class docstring
    and ``docs/eval_schemes.md``.

    Exists so ``train.py`` doesn't need a bespoke sibling script (e.g. the
    deleted ``single_split_eval.py``) just to get pooled-CI evaluation: any
    ``train.py`` run whose ``data`` config builds ``data_val``/``data_test``
    with ``return_index=True`` (``IndexedHealthDataModule`` or
    ``WalkForwardHealthDataModule``) can attach this callback instead and
    get the same treatment, with ``train.py``/``train.yaml`` themselves
    unmodified.

    Internally composes a ``PredictionCollectorCallback`` (delegates
    ``on_test_batch_end`` to it) rather than duplicating its batch-index-to-
    source-row join logic, then on ``on_test_end`` computes pooled metrics
    over everything that callback collected and writes them alongside the
    saved prediction table.

    Not a fit for walk-forward CV (``WalkForwardEvalPlan``): that plan pools
    predictions across *multiple* ``trainer.test()`` calls (one per fold)
    into a single evaluation set, which no single ``Callback`` instance's
    ``on_test_end`` can do -- it only ever sees one trainer's one test run.
    ``WalkForwardEvalPlan`` handles its own multi-fold orchestration, sharing the
    pure pooled-metric functions in ``src/utils/pooled_metrics.py`` with
    this class rather than duplicating them.
    """

    def __init__(self, output_filename: str = "pooled_predictions.csv", n_bootstraps: int = 1000) -> None:
        """Initializes the PooledMetricsCallback.

        Args:
            output_filename: Name of the CSV written to the trainer's
                ``default_root_dir`` (Hydra's per-run output directory in
                normal use) containing every pooled test prediction.
            n_bootstraps: BCa bootstrap resamples per metric. See
                ``compute_pooled_metrics`` for the precision/stability
                tradeoff at higher values.
        """
        super().__init__()
        self.n_bootstraps = n_bootstraps
        self.output_filename = output_filename
        self._collector = PredictionCollectorCallback()
        self.pooled_metrics: dict = {}
        self.pooled_predictions: pd.DataFrame = pd.DataFrame()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._collector.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        test_df = self._collector.to_dataframe()
        pooled_df = test_df[test_df["stage"] == "test"].copy()
        self.pooled_predictions = pooled_df

        classification_metrics_callback = next(
            (cb for cb in trainer.callbacks if type(cb).__name__ == "ClassificationMetricsCallback"),
            None,
        )
        self.pooled_metrics = compute_pooled_metrics(
            pooled_df, log,
            classification_metrics_callback=classification_metrics_callback,
            n_bootstraps=self.n_bootstraps,
        )

        output_dir = getattr(trainer, "default_root_dir", None)
        if output_dir and not pooled_df.empty:
            import os
            out_path = os.path.join(output_dir, self.output_filename)
            os.makedirs(output_dir, exist_ok=True)
            pooled_df.to_csv(out_path, index=False)
            log.info(f"Saved pooled predictions ({len(pooled_df)} rows) to {out_path}")

        for l in trainer.loggers:
            if hasattr(l, "log_metrics"):
                l.log_metrics(self.pooled_metrics)
