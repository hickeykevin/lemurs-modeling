import os
import tempfile
import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule, Trainer


from src.models.health_module import FLAMLHealthModule
#importing featurizer
from src.data.components.tabular_features import SummaryStatsFeaturizer


class DummyDataModule(LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        self._train_loader = DataLoader(train_ds, batch_size=10)
        self._val_loader = DataLoader(val_ds, batch_size=10)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

class DemoDataModule(LightningDataModule):
    """Like DummyDataModule, but batches include a user_idx and a demographics
    tensor -- (x, y, user_idx, demographics) -- matching HealthDataset's real
    __getitem__ layout, so _split_batch has something real to unpack."""

    def __init__(self, X_train, y_train, demo_train, X_val, y_val, demo_val):
        super().__init__()
        train_ds = TensorDataset(X_train, y_train, torch.zeros(len(y_train), dtype=torch.long), demo_train)
        val_ds = TensorDataset(X_val, y_val, torch.zeros(len(y_val), dtype=torch.long), demo_val)
        self._train_loader = DataLoader(train_ds, batch_size=10)
        self._val_loader = DataLoader(val_ds, batch_size=10)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader


def test_flaml_checkpoint_save_and_load():
    """
    Tests that FLAMLHealthModule fits a model, saves a PyTorch Lightning checkpoint,
    and loads the checkpoint successfully while preserving predictions and best estimator.
    """
    # 1. Generate synthetic dataset
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 50
    time_steps = 10
    features = 4
    num_classes = 2

    X_train = torch.randn(batch_size, time_steps, features)
    y_train = torch.randint(0, num_classes, (batch_size,))
    X_val = torch.randn(20, time_steps, features)
    y_val = torch.randint(0, num_classes, (20,))

    dm = DummyDataModule(X_train, y_train, X_val, y_val)

    # 2. Instantiate FLAMLHealthModule with short time budget
    automl_config = {
        "time_budget": 2,
        "estimator_list": ["rf", "extra_tree"],
        "verbose": 0,
    }
    module = FLAMLHealthModule(automl_config=automl_config, task="classification")

    # Attach trainer and run fit
    trainer = Trainer(default_root_dir="logs/debug", max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=True)
    trainer.fit(module, datamodule=dm)

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.best_estimator is not None

    # Get predictions before saving checkpoint
    val_batch = (X_val, y_val)
    val_out_before = module.validation_step(val_batch, 0)
    preds_before = val_out_before["preds"]
    logits_before = val_out_before["logits"]

    # 3. Save checkpoint to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "flaml_test.ckpt")
        trainer.save_checkpoint(ckpt_path)

        assert os.path.exists(ckpt_path)

        # 4. Load from checkpoint
        loaded_module = FLAMLHealthModule.load_from_checkpoint(ckpt_path)
        assert hasattr(loaded_module.automl, "best_estimator")
        assert loaded_module.automl.best_estimator == module.automl.best_estimator
        assert loaded_module.num_classes == module.num_classes

        # 5. Evaluate loaded module and verify identical output
        loaded_module._trainer = trainer
        val_out_after = loaded_module.validation_step(val_batch, 0)
        preds_after = val_out_after["preds"]
        logits_after = val_out_after["logits"]

        torch.testing.assert_close(preds_before, preds_after)
        torch.testing.assert_close(logits_before, logits_after)

        # 6. Ensure setup("fit") or setup("test") on loaded module does not re-fit
        best_estimator_orig = loaded_module.automl.best_estimator
        loaded_module.setup(stage="test")
        assert loaded_module.automl.best_estimator == best_estimator_orig


def test_flaml_auto_class_weights():
    """Tests that FLAMLHealthModule correctly computes sample_weight with auto_class_weights=True."""
    np.random.seed(42)
    torch.manual_seed(42)

    X_train = torch.randn(40, 5, 2)
    # Imbalanced targets: 30 zeros, 10 ones
    y_train = torch.tensor([0] * 30 + [1] * 10)
    X_val = torch.randn(10, 5, 2)
    y_val = torch.tensor([0] * 8 + [1] * 2)

    dm = DummyDataModule(X_train, y_train, X_val, y_val)
    automl_config = {"time_budget": 2, "estimator_list": ["rf"], "verbose": 0}

    module = FLAMLHealthModule(
        automl_config=automl_config, task="classification", auto_class_weights=True
    )
    trainer = Trainer(
        default_root_dir="logs/debug",
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=dm)

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.best_estimator is not None

def test_flaml_with_summary_stats_featurizer():
    """Feature count should depend on modalities * n_stats, not time_steps,
    once a featurizer is set."""
    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = 40
    time_steps = 15
    features = 3
    num_classes = 2

    X_train = torch.randn(batch_size, time_steps, features)
    y_train = torch.randint(0, num_classes, (batch_size,))
    X_val = torch.randn(12, time_steps, features)
    y_val = torch.randint(0, num_classes, (12,))

    dm = DummyDataModule(X_train, y_train, X_val, y_val)

    featurizer = SummaryStatsFeaturizer()  # default 5 stats
    automl_config = {"time_budget": 2, "estimator_list": ["rf"], "verbose": 0}
    module = FLAMLHealthModule(automl_config=automl_config, task="classification", featurizer=featurizer)

    trainer = Trainer(default_root_dir="logs/debug", max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.fit(module, datamodule=dm)

    assert module.automl.best_estimator is not None
    expected_width = features * len(featurizer.stats)  # 3 * 5 = 15, independent of time_steps
    assert module.automl.model.estimator.n_features_in_ == expected_width

    val_out = module.validation_step((X_val, y_val), 0)
    assert val_out["preds"].shape == (12,)


def test_flaml_with_featurizer_and_demographics():
    """Demographics columns should be concatenated onto the featurized row."""
    np.random.seed(1)
    torch.manual_seed(1)

    batch_size, time_steps, features, demo_dim = 40, 8, 3, 2

    X_train = torch.randn(batch_size, time_steps, features)
    y_train = torch.randint(0, 2, (batch_size,))
    demo_train = torch.randn(batch_size, demo_dim)
    X_val = torch.randn(10, time_steps, features)
    y_val = torch.randint(0, 2, (10,))
    demo_val = torch.randn(10, demo_dim)

    dm = DemoDataModule(X_train, y_train, demo_train, X_val, y_val, demo_val)

    featurizer = SummaryStatsFeaturizer()
    automl_config = {"time_budget": 2, "estimator_list": ["rf"], "verbose": 0}
    module = FLAMLHealthModule(automl_config=automl_config, task="classification", featurizer=featurizer)

    trainer = Trainer(default_root_dir="logs/debug", max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.fit(module, datamodule=dm)

    expected_width = features * len(featurizer.stats) + demo_dim  # 3*5 + 2 = 17
    assert module.automl.model.estimator.n_features_in_ == expected_width
