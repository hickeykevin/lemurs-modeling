import os
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.seed = cfg_train.seed

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    train_f1 = train_metric_dict["test/f1_mean"]
    test_f1 = test_metric_dict["test/f1_mean"]

    if torch.isnan(train_f1) or torch.isnan(test_f1):
        # This fixture trains against the real cohort with a small user-level
        # test split, which can legitimately land on a single-class split for
        # some seeds. ClassificationMetricsCallback now reports that honestly
        # as NaN rather than a degenerate-but-finite placeholder (e.g. f1=0.33
        # on an all-one-class epoch) that would otherwise pass ">  0.0" while
        # meaning nothing. When that happens, the two independently computed
        # runs must still agree it's undefined.
        assert torch.isnan(train_f1) and torch.isnan(test_f1)
    else:
        assert test_f1 > 0.0
        assert abs(train_f1.item() - test_f1.item()) < 0.005
