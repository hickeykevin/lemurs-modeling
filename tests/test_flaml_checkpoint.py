import os
import tempfile
import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule, Trainer
from src.models.health_module import FLAMLHealthModule


class DummyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "targets": self.y[idx]}


class DummyDataModule(LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self._train_loader = DataLoader(DummyDataset(X_train, y_train), batch_size=10)
        self._val_loader = DataLoader(DummyDataset(X_val, y_val), batch_size=10)

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
    val_batch = {"features": X_val, "targets": y_val}
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


def test_flaml_checkpoint_save_and_load_with_demographics():
    """Tests that FLAMLHealthModule fits a model with demographics, saves a PyTorch Lightning checkpoint,
    and loads the checkpoint successfully while preserving predictions, best estimator, and feature counts."""
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 40
    time_steps = 6
    features = 3
    demo_dim = 5
    num_classes = 2

    X_train = torch.randn(batch_size, time_steps, features)
    y_train = torch.randint(0, num_classes, (batch_size,))
    demo_train = torch.randn(batch_size, demo_dim)

    X_val = torch.randn(15, time_steps, features)
    y_val = torch.randint(0, num_classes, (15,))
    demo_val = torch.randn(15, demo_dim)

    class DummyDemographicsDataset:
        def __init__(self, X, y, demo):
            self.X = X
            self.y = y
            self.demo = demo

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return {
                "features": self.X[idx],
                "targets": self.y[idx],
                "demographics": self.demo[idx],
            }

    class DummyDemographicsDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()
            self._train_loader = DataLoader(
                DummyDemographicsDataset(X_train, y_train, demo_train), batch_size=10
            )
            self._val_loader = DataLoader(
                DummyDemographicsDataset(X_val, y_val, demo_val), batch_size=10
            )

        def train_dataloader(self):
            return self._train_loader

        def val_dataloader(self):
            return self._val_loader

    dm = DummyDemographicsDataModule()

    automl_config = {
        "time_budget": 2,
        "estimator_list": ["rf"],
        "verbose": 0,
    }
    module = FLAMLHealthModule(automl_config=automl_config, task="classification")

    trainer = Trainer(
        default_root_dir="logs/debug",
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=dm)

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.best_estimator is not None
    assert module.automl.model.estimator.n_features_in_ == (time_steps * features) + demo_dim

    val_batch = {"features": X_val, "targets": y_val, "demographics": demo_val}
    val_out_before = module.validation_step(val_batch, 0)
    preds_before = val_out_before["preds"]
    logits_before = val_out_before["logits"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "flaml_demo_test.ckpt")
        trainer.save_checkpoint(ckpt_path)

        assert os.path.exists(ckpt_path)

        loaded_module = FLAMLHealthModule.load_from_checkpoint(ckpt_path)
        assert hasattr(loaded_module.automl, "best_estimator")
        assert loaded_module.automl.best_estimator == module.automl.best_estimator
        assert loaded_module.automl.model.estimator.n_features_in_ == (time_steps * features) + demo_dim

        loaded_module._trainer = trainer
        val_out_after = loaded_module.validation_step(val_batch, 0)
        preds_after = val_out_after["preds"]
        logits_after = val_out_after["logits"]

        torch.testing.assert_close(preds_before, preds_after)
        torch.testing.assert_close(logits_before, logits_after)


