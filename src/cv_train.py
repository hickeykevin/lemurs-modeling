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

    test_metrics_all_folds = defaultdict(list)
    object_dict_for_logging = {"cfg": cfg, "logger": logger}

    for fold in range(num_folds):
        log.info(f"--- Starting fold {fold + 1}/{num_folds} ---")

        with open_dict(cfg):
            cfg.data.current_fold = fold

        log.info(f"Instantiating datamodule <{cfg.data._target_}> for fold {fold}")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        if fold == 0 and logger:
            log.info("Logging hyperparameters!")
            object_dict_for_logging.update({
                "datamodule": datamodule,
                "model": model,
                "callbacks": callbacks,
                "trainer": trainer,
            })
            log_hyperparameters(object_dict_for_logging)

        if cfg.get("train"):
            log.info(f"Starting training for fold {fold}!")
            trainer.fit(model=model, datamodule=datamodule)

        if cfg.get("test"):
            log.info(f"Starting testing for fold {fold}!")
            ckpt_path = trainer.checkpoint_callback.best_model_path if getattr(trainer, 'checkpoint_callback', None) else ""
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path for fold {fold}: {ckpt_path}")

        test_metrics = trainer.callback_metrics

        # Collect and prefix test metrics for this fold
        for k, v in test_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            test_metrics_all_folds[k].append(val)

            # Log the specific fold metric
            for l in logger:
                if hasattr(l, "log_metrics"):
                    l.log_metrics({f"{k}_fold_{fold}": val})

    # Aggregate test metrics
    agg_metrics = {}
    for k, v_list in test_metrics_all_folds.items():
        mean_val = float(np.mean(v_list))
        std_val = float(np.std(v_list))
        agg_metrics[f"{k}_mean"] = mean_val
        agg_metrics[f"{k}_std"] = std_val

        # Log aggregated metrics
        for l in logger:
            if hasattr(l, "log_metrics"):
                l.log_metrics({f"{k}_mean": mean_val, f"{k}_std": std_val})

        log.info(f"{k} across {num_folds} folds: {mean_val:.4f} ± {std_val:.4f}")

    return agg_metrics, object_dict_for_logging


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
