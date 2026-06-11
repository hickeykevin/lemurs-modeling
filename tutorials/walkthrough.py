import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# Setup the project root
# This adds the project root to the python path so imports like `from src import utils` work.
# It also loads environment variables from .env.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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

# Register OmegaConf resolvers (like len resolver)
register_resolvers()


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model step-by-step with interactive breakpoints.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print("\n" + "="*80)
    print("🎯 STARTING INTERACTIVE HYDRA & LIGHTNING WALKTHROUGH")
    print("="*80)
    print(f"Active Config:\n{OmegaConf.to_yaml(cfg)}")
    print("="*80 + "\n")

    # --- Datamodule Instantiation ---
    print("🧱 STEP 1: Datamodule Instantiation")
    print("About to instantiate the datamodule.")
    print(f"Config section (cfg.data):\n{OmegaConf.to_yaml(cfg.data)}")
    print(">>> [PDB BREAKPOINT] Step in or inspect 'cfg.data'. Type 'c' or 'continue' to proceed.")
    import pdb; pdb.set_trace()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    print(f"\n✅ Datamodule instantiated successfully!")
    print(f"Class type: {type(datamodule)}")
    print(">>> [PDB BREAKPOINT] Inspect 'datamodule' or 'datamodule.hparams'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    # --- Datamodule Setup ---
    print("\n⚙️ STEP 1.5: Datamodule Setup")
    print("About to run 'datamodule.setup(stage=\"fit\")'. This connects to the database and builds splits.")
    print(">>> [PDB BREAKPOINT] Step in/inspect 'datamodule.setup'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    datamodule.setup(stage="fit")

    print(f"\n✅ Datamodule setup complete!")
    if hasattr(datamodule, "data_train") and datamodule.data_train is not None:
        print(f"Train dataset size: {len(datamodule.data_train)}")
        print(f"Validation dataset size: {len(datamodule.data_val)}")
        print(f"Test dataset size: {len(datamodule.data_test)}")
        # Inspect a single sample shape
        try:
            feat, target = datamodule.data_train[0]
            print(f"Sample sequence shape: {feat.shape} [Sequence length, Features]")
            print(f"Sample target label: {target}")
        except Exception as e:
            print(f"Could not retrieve sample: {e}")

    # --- Model Instantiation ---
    print("\n🧠 STEP 2: Model Instantiation")
    print("About to instantiate the model.")
    print(f"Config section (cfg.model):\n{OmegaConf.to_yaml(cfg.model)}")
    print(">>> [PDB BREAKPOINT] Step in/inspect 'cfg.model' (note the nested cfg.model.net configuration). Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    print(f"\n✅ Model instantiated successfully!")
    print(f"Class type: {type(model)}")
    print(f"Underlying network (model.net) type: {type(model.net)}")
    print(">>> [PDB BREAKPOINT] Inspect 'model' or 'model.net'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    # --- Callbacks Instantiation ---
    print("\n🔔 STEP 3: Callbacks Instantiation")
    print("About to instantiate callbacks.")
    print(f"Config section (cfg.callbacks):\n{OmegaConf.to_yaml(cfg.callbacks)}")
    print(">>> [PDB BREAKPOINT] Step in/inspect 'cfg.callbacks'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    print(f"\n✅ Callbacks instantiated successfully!")
    print(f"Instantiated callbacks: {[type(cb).__name__ for cb in callbacks]}")
    print(">>> [PDB BREAKPOINT] Inspect 'callbacks'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    # --- Loggers Instantiation ---
    print("\n📝 STEP 4: Loggers Instantiation")
    print("About to instantiate loggers.")
    print(f"Config section (cfg.logger):\n{OmegaConf.to_yaml(cfg.logger)}")
    print(">>> [PDB BREAKPOINT] Step in/inspect 'cfg.logger'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    print(f"\n✅ Loggers instantiated successfully!")
    print(f"Instantiated loggers: {[type(lg).__name__ for lg in logger]}")
    print(">>> [PDB BREAKPOINT] Inspect 'logger'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    # --- Trainer Instantiation ---
    print("\n⚡ STEP 5: Trainer Instantiation")
    print("About to instantiate the PyTorch Lightning Trainer.")
    print(f"Config section (cfg.trainer):\n{OmegaConf.to_yaml(cfg.trainer)}")
    print(">>> [PDB BREAKPOINT] Step in/inspect 'cfg.trainer'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    print(f"\n✅ Trainer instantiated successfully!")
    print(f"Class type: {type(trainer)}")
    print(">>> [PDB BREAKPOINT] Inspect 'trainer'. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # --- Training Loop ---
    if cfg.get("train"):
        print("\n🚀 STEP 6: Starting Training Loop")
        print("About to call 'trainer.fit(model, datamodule)' to start the fit loop.")
        print(">>> [PDB BREAKPOINT] Step in/inspect 'trainer.fit'. Type 'c' to proceed.")
        import pdb; pdb.set_trace()

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # --- Testing Loop ---
    if cfg.get("test"):
        print("\n🧪 STEP 7: Starting Testing Loop")
        print("About to call 'trainer.test(model, datamodule)' with the best checkpoint.")
        print(">>> [PDB BREAKPOINT] Step in/inspect 'trainer.test'. Type 'c' to proceed.")
        import pdb; pdb.set_trace()

        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    print("\n🎉 Walkthrough successfully completed!")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training walkthrough.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
