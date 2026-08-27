import os
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

from src.utils.checkpoint_compat import allow_full_checkpoint_loading
from src.utils.prediction_collector import PredictionCollectorCallback
from src.wf_cv_train import _compute_pooled_metrics, _save_pooled_predictions
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

allow_full_checkpoint_loading()


@task_wrapper
def single_split_eval(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains and evaluates one single-split run (e.g. split_mode="longitudinal"),
    reporting the SAME pooled, user-cluster BCa bootstrap metrics wf_cv_train.py
    reports for its CV schemes, instead of ClassificationMetricsCallback's
    row-level torchmetrics.BootStrapper CI.

    Exists specifically so a single-split baseline (one train/val/test cut,
    no CV folds) can be compared against wf_cv_train.py's walk-forward
    schemes on equal footing: a row-level bootstrap understates uncertainty
    here for the same reason it is wrong everywhere else in this pipeline
    (see CohortSplitter's class docstring) -- resampling by app_user_id,
    not by row, is what actually matches this cohort's source of variance.

    Requires cfg.data to instantiate a datamodule whose data_val/data_test
    are built with return_index=True (e.g. IndexedHealthDataModule) so
    PredictionCollectorCallback can collect predictions to pool.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with pooled metrics and a dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    collector = PredictionCollectorCallback()
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks + [collector], logger=logger
    )

    object_dict = {
        "cfg": cfg, "datamodule": datamodule, "model": model,
        "callbacks": callbacks, "logger": logger, "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path if getattr(trainer, "checkpoint_callback", None) else ""
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    log.info(f"Best ckpt path: {ckpt_path}")

    test_df = collector.to_dataframe()
    pooled_df = test_df[test_df["stage"] == "test"].copy()

    pooled_metrics = _compute_pooled_metrics(pooled_df, log, cfg=cfg)
    _save_pooled_predictions(cfg, pooled_df, log)

    object_dict["pooled_predictions"] = pooled_df
    metric_dict = {**trainer.callback_metrics, **pooled_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="single_split_eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for single-split pooled-bootstrap evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with the optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = single_split_eval(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
