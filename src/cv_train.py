import os
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
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

# Per-run rows are logged at a step far above any training step. Loggers that
# assume monotonically increasing steps (W&B, TensorBoard) otherwise drop these
# points, since each fold's trainer restarts its step counter from zero.
CV_LOG_STEP_OFFSET = 1_000_000

# Checkpoints here embed Hydra-instantiated hparams, which torch>=2.6 will not
# unpickle under its weights_only default. Safe: we wrote these files ourselves.
allow_full_checkpoint_loading()


def _user_hash(user_ids: Any) -> float:
    """Hashes a set of user ids to a value that survives a float metric column.

    Loggers only accept scalars, so the fingerprint has to be a number. Eight
    hex digits keeps it under 2^32, which is exactly representable as a float
    and so round-trips through a CSV column without silent rounding.
    """
    key = ",".join(sorted(str(u) for u in set(user_ids)))
    return float(int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16))


def _fold_identity(datamodule: LightningDataModule) -> Dict[str, float]:
    """Fingerprints the cohort and the held-out users of the current fold.

    This is what makes a paired comparison between two configurations checkable
    rather than assumed. The partition for a given (repeat, fold) depends only
    on the seed and the surviving cohort, never on the model — so two runs that
    share a ``cohort_hash`` are splitting the same people, and two runs whose
    folds also share a ``test_user_hash`` held out the identical users and can
    be differenced fold by fold.

    The cohort is *not* invariant to every data setting: anything that changes
    sensor-coverage filtering (a different sampler window, modality set, or
    collapse strategy) drops different responses and so yields a different
    ``cohort_hash``. Logging it means that case surfaces as a mismatch at
    comparison time instead of quietly producing a pairing that isn't one.
    """
    identity: Dict[str, float] = {}

    master_df = getattr(datamodule, "master_df", None)
    if master_df is not None and "app_user_id" in master_df:
        identity["cv/cohort_hash"] = _user_hash(master_df["app_user_id"])
        identity["cv/cohort_n_users"] = float(master_df["app_user_id"].nunique())
        identity["cv/cohort_n_responses"] = float(len(master_df))

    data_test = getattr(datamodule, "data_test", None)
    test_links = getattr(data_test, "data_links", None)
    if test_links is not None and "app_user_id" in test_links:
        identity["cv/test_user_hash"] = _user_hash(test_links["app_user_id"])
        identity["cv/test_n_users"] = float(test_links["app_user_id"].nunique())
        identity["cv/test_n_responses"] = float(len(test_links))

    return identity


@task_wrapper
def cv_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model using Cross Validation.

    This method iterates over folds, instantiating the datamodule, model, and trainer
    for each fold. Finally, it aggregates test metrics across all folds.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with aggregated metrics and a dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    num_folds = cfg.data.get("num_folds", 5)

    if num_folds == -1:
        log.info("Resolving Leave-One-User-Out Cross Validation (-1 folds)...")
        temp_datamodule = hydra.utils.instantiate(cfg.data)
        if hasattr(temp_datamodule, "get_num_folds"):
            num_folds = temp_datamodule.get_num_folds()
            log.info(f"Leave-One-User-Out CV selected. Detected {num_folds} folds/users.")
            with open_dict(cfg):
                cfg.data.num_folds = num_folds
        else:
            raise ValueError("The configured datamodule does not support dynamic fold counting (-1).")

    num_repeats = cfg.data.get("num_repeats", 1)

    test_metrics_all_folds = defaultdict(list)
    per_run_rows: List[Dict[str, float]] = []
    cohort_hashes_seen: set = set()
    object_dict_for_logging = {"cfg": cfg, "logger": logger}

    total_runs = num_repeats * num_folds
    log.info(
        f"Repeated grouped CV: {num_repeats} repeat(s) x {num_folds} fold(s) = {total_runs} runs"
    )

    run = 0
    for repeat in range(num_repeats):
        for fold in range(num_folds):
            run += 1
            log.info(
                f"--- repeat {repeat + 1}/{num_repeats}, fold {fold + 1}/{num_folds} "
                f"(run {run}/{total_runs}) ---"
            )

            with open_dict(cfg):
                cfg.data.current_fold = fold
                cfg.data.current_repeat = repeat

            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
            model: LightningModule = hydra.utils.instantiate(cfg.model)
            callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )

            if run == 1 and logger:
                log.info("Logging hyperparameters!")
                object_dict_for_logging.update({
                    "datamodule": datamodule,
                    "model": model,
                    "callbacks": callbacks,
                    "trainer": trainer,
                })
                log_hyperparameters(object_dict_for_logging)

            if cfg.get("train"):
                trainer.fit(model=model, datamodule=datamodule)

            if cfg.get("test"):
                ckpt_path = trainer.checkpoint_callback.best_model_path if getattr(trainer, 'checkpoint_callback', None) else ""
                if ckpt_path == "":
                    log.warning("Best ckpt not found! Using current weights for testing...")
                    ckpt_path = None
                trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

            # One row per run, tagged with the fold's identity. Emitting the
            # metrics and the fingerprint of the users they were measured on in
            # a single record is what lets a later comparison line two sweeps up
            # by the partition itself rather than trusting the fold index.
            run_row: Dict[str, float] = {
                "cv/repeat": float(repeat),
                "cv/fold": float(fold),
                "cv/run": float(run),
            }
            run_row.update(_fold_identity(datamodule))

            for k, v in trainer.callback_metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                test_metrics_all_folds[k].append(val)
                run_row[f"fold/{k}"] = val

            per_run_rows.append(run_row)

            cohort_hash = run_row.get("cv/cohort_hash")
            if cohort_hash is not None:
                if cohort_hashes_seen and cohort_hash not in cohort_hashes_seen:
                    log.warning(
                        "Cohort changed between runs within this sweep — folds are "
                        "not drawn from a fixed set of users, so the aggregate below "
                        "mixes partitions of different cohorts."
                    )
                cohort_hashes_seen.add(cohort_hash)

            for l in logger:
                if hasattr(l, "log_metrics"):
                    l.log_metrics(run_row, step=CV_LOG_STEP_OFFSET + run)

    agg_metrics = _aggregate_cv_metrics(test_metrics_all_folds, logger, log, total_runs)

    # The last trainer has already torn down by this point, so nothing else will
    # flush the per-run and aggregate rows written above; without this they can
    # be lost entirely for buffered loggers such as CSVLogger.
    for l in logger:
        if hasattr(l, "save"):
            l.save()

    object_dict_for_logging["per_run_metrics"] = per_run_rows
    return agg_metrics, object_dict_for_logging


def _aggregate_cv_metrics(
    metrics_by_run: Dict[str, List[float]],
    logger: List[Logger],
    log: RankedLogger,
    total_runs: int,
) -> Dict[str, Any]:
    """Summarises per-run metrics as mean, spread, and a 95% interval.

    Reporting a bare mean over folds hides the thing that matters most at this
    cohort size: with ~31 users and roughly a dozen ever positive, which users
    land in a given test fold moves the result more than most modelling choices
    do. The spread and interval are therefore reported as first-class outputs,
    not diagnostics — a mean AUROC quoted without them is not interpretable.

    NaNs are excluded rather than propagated, since a fold whose test users are
    all one class yields an undefined AUROC that should not erase the others.
    """
    agg_metrics: Dict[str, Any] = {}
    logged: Dict[str, float] = {}

    table_rows = []
    for k, v_list in sorted(metrics_by_run.items()):
        values = np.asarray([v for v in v_list if v is not None and not np.isnan(v)], dtype=float)
        n = len(values)
        if n == 0:
            log.warning(f"{k}: no finite values across {total_runs} runs")
            continue

        mean_val = float(values.mean())
        # Sample standard deviation: these runs are a sample of possible splits,
        # not the population of them.
        std_val = float(values.std(ddof=1)) if n > 1 else 0.0
        stderr = std_val / np.sqrt(n) if n > 1 else 0.0
        ci_low, ci_high = mean_val - 1.96 * stderr, mean_val + 1.96 * stderr

        agg_metrics[f"{k}_mean"] = mean_val
        agg_metrics[f"{k}_std"] = std_val
        agg_metrics[f"{k}_ci_low"] = float(ci_low)
        agg_metrics[f"{k}_ci_high"] = float(ci_high)
        agg_metrics[f"{k}_n_runs"] = float(n)

        # Include cv_summary/ prefixed entries in agg_metrics as well
        agg_metrics[f"cv_summary/{k}_mean"] = mean_val
        agg_metrics[f"cv_summary/{k}_std"] = std_val
        agg_metrics[f"cv_summary/{k}_ci_low"] = float(ci_low)
        agg_metrics[f"cv_summary/{k}_ci_high"] = float(ci_high)
        agg_metrics[f"cv_summary/{k}_n_runs"] = float(n)

        logged.update({
            f"cv_summary/{k}_mean": mean_val,
            f"cv_summary/{k}_std": std_val,
            f"cv_summary/{k}_ci_low": float(ci_low),
            f"cv_summary/{k}_ci_high": float(ci_high),
            f"cv_summary/{k}_n_runs": float(n),
        })

        skipped = "" if n == total_runs else f" [{total_runs - n} undefined]"
        table_rows.append([
            k,
            f"{mean_val:.4f} ± {std_val:.4f}",
            f"[{ci_low:.4f}, {ci_high:.4f}]",
            f"{n}{skipped}",
        ])

        log.info(
            f"{k}: {mean_val:.4f} +/- {std_val:.4f} (sd)  "
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  n={n}{skipped}"
        )

    # Print a Rich terminal table summary
    try:
        from rich.table import Table
        from rich.console import Console

        rich_table = Table(
            title=f"Cross-Validation Summary ({total_runs} runs)",
            show_header=True,
            header_style="bold cyan",
        )
        rich_table.add_column("Metric", style="bold white")
        rich_table.add_column("Mean ± SD", justify="center")
        rich_table.add_column("95% CI", justify="center")
        rich_table.add_column("N Runs", justify="right")

        for row in table_rows:
            rich_table.add_row(*row)

        console = Console()
        console.print(rich_table)
    except Exception:
        pass

    # Log to WandB as an interactive table if WandbLogger is active
    try:
        from lightning.pytorch.loggers.wandb import WandbLogger
        import wandb

        wandb_table = wandb.Table(
            columns=["Metric", "Mean ± SD", "95% CI", "N Runs"],
            data=table_rows,
        )
        for l in logger:
            if isinstance(l, WandbLogger) and hasattr(l, "experiment") and l.experiment is not None:
                l.experiment.log({"cv_summary/table": wandb_table})
    except Exception:
        pass

    # A single row after the per-run rows, so the summary sorts last and cannot
    # be mistaken for another fold.
    for l in logger:
        if hasattr(l, "log_metrics"):
            l.log_metrics(logged, step=CV_LOG_STEP_OFFSET + total_runs + 1)

    return agg_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="cv_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for CV training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = cv_train(cfg)

    metric_name = cfg.get("optimized_metric")
    if metric_name:
        metric_name = f"{metric_name}_mean"

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=metric_name
    )

    return metric_value


if __name__ == "__main__":
    main()
