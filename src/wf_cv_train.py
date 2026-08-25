import os
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

from src.utils.checkpoint_compat import allow_full_checkpoint_loading
from src.utils.prediction_collector import PredictionCollectorCallback
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    register_resolvers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()

# See cv_train.py's CV_LOG_STEP_OFFSET for why: each fold's trainer restarts
# its own step counter from zero, and a step-monotonic logger (W&B,
# TensorBoard) drops a per-run row logged at a colliding step otherwise.
WF_LOG_STEP_OFFSET = 1_000_000

allow_full_checkpoint_loading()


@task_wrapper
def wf_cv_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains and evaluates the model across every walk-forward fold, pooling predictions.

    Unlike ``cv_train.py``'s repeated grouped k-fold (which answers "does this
    generalize to a new user with no history?" and so reports fold metrics
    averaged with a spread), this answers "given a user's own history, can
    the model forecast their near-future risk?" — a question best answered by
    pooling every fold's out-of-fold test predictions into one evaluation set
    rather than averaging per-fold scores. See
    ``WalkForwardHealthDataModule`` and ``CohortSplitter.split_walk_forward``
    for why folds are not directly comparable to each other the way grouped
    CV folds are (fold index is a per-user position, not a shared calendar
    window), which is exactly why pooling — not fold-averaging — is the
    primary aggregation here. A secondary per-fold breakdown is still
    reported (see ``_fold_breakdown_table``), to see whether forecasting
    improves as more history accumulates, but it is a diagnostic, not the
    headline number.

    There is no repeat axis here (unlike ``cv_train.py``'s ``num_repeats``):
    a walk-forward fold sequence is not an independently-reshuffled draw the
    way a grouped k-fold's fold assignment is, so re-running it with a
    different seed would not answer a new question — see
    ``WalkForwardHealthDataModule``'s docstring.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with pooled metrics and a dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Fold count depends on the cohort's actual response-count distribution
    # (see CohortSplitter.split_walk_forward), so it is resolved dynamically,
    # the same way cv_train.py resolves num_folds=-1 (leave-one-user-out).
    # get_num_folds() runs setup() as a side effect (it needs master_df to
    # count folds), so this first datamodule already holds the built cohort
    # afterward -- get_prebuilt_cohort() below reuses it for every fold
    # instead of re-hitting the database each time.
    log.info("Resolving walk-forward fold count from the cohort's response counts...")
    first_datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    if not hasattr(first_datamodule, "get_num_folds"):
        raise ValueError(
            "cfg.data must instantiate a datamodule with get_num_folds() "
            "(e.g. WalkForwardHealthDataModule) for wf_cv_train.py."
        )
    num_folds = first_datamodule.get_num_folds()
    if num_folds == 0:
        raise ValueError(
            "This cohort/config produces zero walk-forward folds -- lower "
            "burn_in_responses/val_responses/step_responses, or check the "
            "cohort's response-count distribution."
        )
    log.info(f"Walk-forward CV: {num_folds} fold(s), no repeats (see docstring).")

    shared_cohort = (
        first_datamodule.get_prebuilt_cohort()
        if hasattr(first_datamodule, "get_prebuilt_cohort") else None
    )

    per_run_rows: List[Dict[str, float]] = []
    pooled_predictions: List[pd.DataFrame] = []
    object_dict_for_logging = {"cfg": cfg, "logger": logger}

    for fold in range(num_folds):
        log.info(f"--- fold {fold + 1}/{num_folds} ---")

        with open_dict(cfg):
            cfg.data.current_fold = fold

        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.data, prebuilt_cohort=shared_cohort
        )
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        collector = PredictionCollectorCallback(fold_index=fold)
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks + [collector], logger=logger
        )

        if fold == 0 and logger:
            log.info("Logging hyperparameters!")
            object_dict_for_logging.update({
                "datamodule": datamodule, "model": model,
                "callbacks": callbacks, "trainer": trainer,
            })
            log_hyperparameters(object_dict_for_logging)

        if cfg.get("train"):
            trainer.fit(model=model, datamodule=datamodule)

        # Unlike cv_train.py, testing here is not gated on cfg.get("test"):
        # pooled test predictions are the entire point of this script, not
        # an optional extra, so skipping it would leave nothing to pool.
        ckpt_path = trainer.checkpoint_callback.best_model_path if getattr(trainer, "checkpoint_callback", None) else ""
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        fold_df = collector.to_dataframe()
        fold_test_df = fold_df[fold_df["stage"] == "test"].copy()
        pooled_predictions.append(fold_test_df)

        run_row: Dict[str, float] = {"cv/fold": float(fold), "cv/n_test_rows": float(len(fold_test_df))}
        for k, v in trainer.callback_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            run_row[f"fold/{k}"] = val
        per_run_rows.append(run_row)

        for l in logger:
            if hasattr(l, "log_metrics"):
                l.log_metrics(run_row, step=WF_LOG_STEP_OFFSET + fold)

    pooled_df = pd.concat(pooled_predictions, ignore_index=True) if pooled_predictions else pd.DataFrame()
    pooled_metrics = _compute_pooled_metrics(pooled_df, log)
    fold_breakdown = _fold_breakdown_table(pooled_df, log)

    for l in logger:
        if hasattr(l, "log_metrics"):
            l.log_metrics(pooled_metrics, step=WF_LOG_STEP_OFFSET + num_folds + 1)
        if hasattr(l, "save"):
            l.save()

    # Persist the full pooled prediction table alongside the run's own logs,
    # not just the summary metrics -- this is what a paper's pooled AUROC/CI
    # is computed from, and it should be reproducible/inspectable afterward.
    _save_pooled_predictions(cfg, pooled_df, log)

    object_dict_for_logging["per_run_metrics"] = per_run_rows
    object_dict_for_logging["pooled_predictions"] = pooled_df
    object_dict_for_logging["fold_breakdown"] = fold_breakdown
    return pooled_metrics, object_dict_for_logging


def _compute_pooled_metrics(pooled_df: pd.DataFrame, log: RankedLogger) -> Dict[str, float]:
    """Pooled AUROC/AUPRC over every fold's out-of-fold test predictions, with a
    participant-cluster bootstrap CI.

    Resampling by app_user_id, not by row, matches this cohort's actual
    source of variance: positives are concentrated in a handful of
    participants (see CohortSplitter's class docstring), so a row-level
    bootstrap would understate uncertainty by treating each of one user's
    many responses as an independent draw.
    """
    metrics: Dict[str, float] = {}
    if pooled_df.empty:
        log.warning("Pooled prediction table is empty; no folds produced test predictions.")
        return metrics

    prob_cols = sorted(c for c in pooled_df.columns if c.startswith("prob_class_"))
    if len(prob_cols) != 2:
        log.warning(
            f"wf_cv_train's pooled metrics currently assume binary classification "
            f"(found {len(prob_cols)} probability columns); skipping AUROC/AUPRC."
        )
        return metrics

    y_true = pooled_df["y_true"].to_numpy()
    y_score = pooled_df[prob_cols[1]].to_numpy()  # P(class 1)

    if len(np.unique(y_true)) < 2:
        log.warning("Pooled test predictions contain only one class; AUROC/AUPRC undefined.")
        return metrics

    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics["pooled/auroc"] = float(roc_auc_score(y_true, y_score))
    metrics["pooled/auprc"] = float(average_precision_score(y_true, y_score))
    metrics["pooled/n_predictions"] = float(len(pooled_df))
    metrics["pooled/n_users"] = float(pooled_df["app_user_id"].nunique())
    metrics["pooled/n_positive"] = float(y_true.sum())

    ci = _cluster_bootstrap_ci(pooled_df, prob_cols[1], n_bootstraps=1000, seed=0)
    metrics.update(ci)

    log.info(
        f"Pooled AUROC: {metrics['pooled/auroc']:.4f}  "
        f"(95% user-cluster bootstrap CI [{ci['pooled/auroc_ci_low']:.4f}, "
        f"{ci['pooled/auroc_ci_high']:.4f}], n={int(metrics['pooled/n_predictions'])} "
        f"predictions from {int(metrics['pooled/n_users'])} users)"
    )
    log.info(
        f"Pooled AUPRC: {metrics['pooled/auprc']:.4f}  "
        f"(95% user-cluster bootstrap CI [{ci['pooled/auprc_ci_low']:.4f}, "
        f"{ci['pooled/auprc_ci_high']:.4f}])"
    )
    return metrics


def _cluster_bootstrap_ci(
    pooled_df: pd.DataFrame, prob_col: str, n_bootstraps: int, seed: int
) -> Dict[str, float]:
    """Percentile bootstrap CI for pooled AUROC/AUPRC, resampling whole users with replacement.

    Each bootstrap replicate draws a set of users (with replacement, same
    size as the original user count) and pools every one of their
    predictions across every fold — matching the "resample the population"
    logic a cluster bootstrap requires, rather than resampling individual
    rows independently of which user they came from.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    users = pooled_df["app_user_id"].unique()
    rng = np.random.RandomState(seed)

    aurocs, auprcs = [], []
    for _ in range(n_bootstraps):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        rows = pd.concat([pooled_df[pooled_df["app_user_id"] == u] for u in sampled_users], ignore_index=True)
        y_true = rows["y_true"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_score = rows[prob_col].to_numpy()
        aurocs.append(roc_auc_score(y_true, y_score))
        auprcs.append(average_precision_score(y_true, y_score))

    def _pct_ci(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    auroc_lo, auroc_hi = _pct_ci(aurocs)
    auprc_lo, auprc_hi = _pct_ci(auprcs)
    return {
        "pooled/auroc_ci_low": auroc_lo,
        "pooled/auroc_ci_high": auroc_hi,
        "pooled/auprc_ci_low": auprc_lo,
        "pooled/auprc_ci_high": auprc_hi,
        "pooled/n_bootstraps_used": float(len(aurocs)),
    }


def _fold_breakdown_table(pooled_df: pd.DataFrame, log: RankedLogger) -> pd.DataFrame:
    """Per-fold AUROC, as a secondary diagnostic for whether forecasting improves
    with more accumulated history -- not the headline result (see module docstring)."""
    if pooled_df.empty:
        return pd.DataFrame()

    from sklearn.metrics import roc_auc_score

    prob_cols = sorted(c for c in pooled_df.columns if c.startswith("prob_class_"))
    if len(prob_cols) != 2:
        return pd.DataFrame()

    rows = []
    for fold_index, group in pooled_df.groupby("fold_index"):
        y_true = group["y_true"].to_numpy()
        auroc = float(roc_auc_score(y_true, group[prob_cols[1]])) if len(np.unique(y_true)) > 1 else float("nan")
        rows.append({
            "fold_index": fold_index,
            "n_predictions": len(group),
            "n_users": group["app_user_id"].nunique(),
            "n_positive": int(y_true.sum()),
            "auroc": auroc,
        })
    table = pd.DataFrame(rows).sort_values("fold_index")

    try:
        from rich.table import Table
        from rich.console import Console

        rich_table = Table(title="Walk-Forward Per-Fold Breakdown (diagnostic, not headline)", show_header=True, header_style="bold cyan")
        for col in table.columns:
            rich_table.add_column(col, justify="right" if col != "fold_index" else "left")
        for _, row in table.iterrows():
            rich_table.add_row(*[
                f"{v:.4f}" if isinstance(v, float) and col == "auroc" else str(v)
                for col, v in zip(table.columns, row)
            ])
        console = Console(record=True)
        console.print(rich_table)
        log.info(f"\n{console.export_text()}")
    except Exception:
        pass

    return table


def _save_pooled_predictions(cfg: DictConfig, pooled_df: pd.DataFrame, log: RankedLogger) -> None:
    """Writes the pooled prediction table to the run's output directory."""
    output_dir = cfg.paths.output_dir if "paths" in cfg and "output_dir" in cfg.paths else None
    if output_dir is None or pooled_df.empty:
        return
    out_path = os.path.join(output_dir, "pooled_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)
    pooled_df.to_csv(out_path, index=False)
    log.info(f"Saved pooled predictions ({len(pooled_df)} rows) to {out_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="wf_cv_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for walk-forward CV training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with the optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = wf_cv_train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
