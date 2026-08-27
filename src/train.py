import os
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

# ------------------------------------------------------------------------------------ #

# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.eval_plans import EvalPlan, run_plan
from src.utils.checkpoint_compat import allow_full_checkpoint_loading
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    register_resolvers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()

# Checkpoints here embed Hydra-instantiated hparams, which torch>=2.6 will not
# unpickle under its weights_only default. Safe: we wrote these files ourselves.
allow_full_checkpoint_loading()


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs this config's evaluation strategy end to end.

    Which strategy runs is chosen by ``cfg.eval_plan`` (a Hydra config group --
    ``eval_plan=single``, ``grouped_cv``, or ``walk_forward``), not by picking a
    different entry script. The plan decides what units of work exist and how
    their results combine; ``run_plan`` executes every plan through one identical
    loop. See ``src/eval_plans/base.py`` and ``docs/eval_schemes.md``.

    ``eval_plan=single`` (the default) is the plain single train/val/test split
    this function ran before the eval_plan layer existed.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    log.info(f"Instantiating eval plan <{cfg.eval_plan._target_}>")
    plan: EvalPlan = hydra.utils.instantiate(cfg.eval_plan)
    return run_plan(cfg, plan)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
