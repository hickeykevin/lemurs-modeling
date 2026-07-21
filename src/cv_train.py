import os
os.environ["OMP_NUM_THREADS"] = "1"

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

# Checkpoints here embed Hydra-instantiated hparams, which torch>=2.6 will not
# unpickle under its weights_only default. Safe: we wrote these files ourselves.
allow_full_checkpoint_loading()


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

            for k, v in trainer.callback_metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                test_metrics_all_folds[k].append(val)

                for l in logger:
                    if hasattr(l, "log_metrics"):
                        l.log_metrics({f"{k}_r{repeat}_f{fold}": val})

    agg_metrics = _aggregate_cv_metrics(test_metrics_all_folds, logger, log, total_runs)
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

        for l in logger:
            if hasattr(l, "log_metrics"):
                l.log_metrics({
                    f"{k}_mean": mean_val,
                    f"{k}_std": std_val,
                    f"{k}_ci_low": float(ci_low),
                    f"{k}_ci_high": float(ci_high),
                })

        skipped = "" if n == total_runs else f"  [{total_runs - n} run(s) undefined]"
        log.info(
            f"{k}: {mean_val:.4f} +/- {std_val:.4f} (sd)  "
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  n={n}{skipped}"
        )

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
