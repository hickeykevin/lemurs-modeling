from unittest.mock import MagicMock
import torch
import pytest
from src.utils.evaluation_callbacks import ConfusionMatrixCallback
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger


def test_confusion_matrix_callback_no_logger():
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 0
    trainer.loggers = []
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act & Assert
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    # This shouldn't raise any errors and should complete successfully
    callback.on_validation_epoch_end(trainer, pl_module)


def test_confusion_matrix_callback_with_wandb():
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 0
    
    # Mock WandbLogger and its experiment property
    wandb_logger = MagicMock(spec=WandbLogger)
    mock_experiment = MagicMock()
    type(wandb_logger).experiment = property(lambda self: mock_experiment)
    
    trainer.loggers = [wandb_logger]
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)
    
    # Assert
    # Verify that logger.experiment.log was called
    mock_experiment.log.assert_called_once()
    
    # Get the logged dict
    logged_dict = mock_experiment.log.call_args[0][0]
    
    assert "val/confusion_matrix" in logged_dict
    assert "val/confusion_matrix_table" in logged_dict
    
    # Verify table columns/data
    table = logged_dict["val/confusion_matrix_table"]
    assert table.columns == ["Actual \\ Predicted", "P0", "P1", "P2"]
    
    # Check data content
    # Predictions: [0, 1, 2, 0], Targets: [0, 1, 1, 2]
    # Matrix shape 3x3:
    # Target 0: prediction 0 (actual class 0 predicted P0) -> 1
    # Target 1: predictions 1, 2 (actual class 1 predicted P1, P2) -> P1: 1, P2: 1
    # Target 2: prediction 0 (actual class 2 predicted P0) -> 1
    # Check actual content:
    # Row 0 (Class 0): [f"Class 0", 1, 0, 0]
    # Row 1 (Class 1): [f"Class 1", 0, 1, 1]
    # Row 2 (Class 2): [f"Class 2", 1, 0, 0]
    assert table.data[0] == ["Class 0", 1, 0, 0]
    assert table.data[1] == ["Class 1", 0, 1, 1]
    assert table.data[2] == ["Class 2", 1, 0, 0]


def test_classification_metrics_callback():
    from src.utils.evaluation_callbacks import ClassificationMetricsCallback
    from unittest.mock import MagicMock
    import torch

    # Arrange
    callback = ClassificationMetricsCallback()
    
    trainer = MagicMock(spec=Trainer)
    trainer.sanity_checking = False
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    pl_module.device = torch.device("cpu")
    
    # Act
    # Simulating validation epoch start, batch end, and epoch end
    callback.on_validation_epoch_start(trainer, pl_module)
    
    # logits shape: (batch_size, num_classes)
    # targets shape: (batch_size)
    outputs = {
        "logits": torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]]),
        "targets": torch.tensor([0, 1])
    }
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)
    
    # Assert
    # Extract logged metrics from pl_module.log calls
    logged_keys = [call[0][0] for call in pl_module.log.call_args_list]
    assert "val/precision" in logged_keys
    assert "val/precision_best" in logged_keys
    assert "val/recall" in logged_keys
    assert "val/recall_best" in logged_keys
    assert "val/f1" in logged_keys
    assert "val/auroc" in logged_keys


def test_confusion_matrix_callback_logs_to_file(tmp_path):
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 2
    trainer.loggers = []
    trainer.log_dir = str(tmp_path)
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)
    
    # Assert
    log_file = tmp_path / "confusion_matrix.txt"
    assert log_file.exists()
    
    content = log_file.read_text()
    assert "=== Confusion Matrix (Validation Epoch 2) ===" in content
    assert "Actual \\ Predicted" in content
    assert "Class 0" in content
    assert "Class 1" in content
    assert "Class 2" in content
    
    # Assert that no ANSI escape sequences (like \x1b or escape brackets) are in the plain text output
    assert "\x1b" not in content
    assert "[3m" not in content
    assert "[1;35m" not in content
    
    # Simulate a second epoch write to verify appending
    trainer.current_epoch = 3
    outputs_epoch3 = {
        "preds": torch.tensor([1, 1, 1]),
        "targets": torch.tensor([1, 1, 1])
    }
    callback.on_validation_batch_end(trainer, pl_module, outputs_epoch3, batch=None, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)
    
    content2 = log_file.read_text()
    assert "=== Confusion Matrix (Validation Epoch 2) ===" in content2
    assert "=== Confusion Matrix (Validation Epoch 3) ===" in content2
    assert "\x1b" not in content2


def test_classification_metrics_callback_test_stage():
    from src.utils.evaluation_callbacks import ClassificationMetricsCallback
    
    # Arrange
    callback = ClassificationMetricsCallback(num_bootstraps=5, sampling_strategy="multinomial")
    
    trainer = MagicMock(spec=Trainer)
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    pl_module.device = torch.device("cpu")
    
    # Act
    # Simulating test batch end and epoch end
    outputs = {
        "logits": torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]]),
        "targets": torch.tensor([0, 1])
    }
    callback.on_test_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, pl_module)
    
    # Assert
    # Extract logged metrics from pl_module.log calls
    logged_keys = [call[0][0] for call in pl_module.log.call_args_list]
    
    # Bootstrapping adds mean, std, and we computed var
    assert "test/f1_mean" in logged_keys
    assert "test/f1_std" in logged_keys
    assert "test/f1_var" in logged_keys
    
    assert "test/auroc_mean" in logged_keys
    assert "test/auroc_std" in logged_keys
    assert "test/auroc_var" in logged_keys
    
    assert "test/precision_mean" in logged_keys
    assert "test/precision_std" in logged_keys
    assert "test/precision_var" in logged_keys


def test_regression_metrics_callback_test_stage(tmp_path):
    from src.utils.evaluation_callbacks import RegressionMetricsCallback
    
    # Arrange
    callback = RegressionMetricsCallback(num_bootstraps=5, sampling_strategy="multinomial")
    
    trainer = MagicMock(spec=Trainer)
    trainer.loggers = []
    trainer.log_dir = str(tmp_path)
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.device = torch.device("cpu")
    
    # Act
    # Simulating test batch end and epoch end
    outputs = {
        "preds": torch.tensor([1.0, 2.0, 3.0]),
        "targets": torch.tensor([1.2, 1.8, 3.1])
    }
    
    callback.on_test_epoch_start(trainer, pl_module)
    callback.on_test_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, pl_module)
    
    # Assert
    logged_keys = [call[0][0] for call in pl_module.log.call_args_list]
    
    # Standard regression metrics
    assert "test/mse_mean" in logged_keys
    assert "test/mse_std" in logged_keys
    assert "test/mse_var" in logged_keys
    
    assert "test/mae_mean" in logged_keys
    assert "test/mae_std" in logged_keys
    assert "test/mae_var" in logged_keys
    
    # Non-minimum bootstrapped metrics
    assert "test/mse_non_min_mean" in logged_keys
    assert "test/mse_non_min_std" in logged_keys
    assert "test/mse_non_min_var" in logged_keys
    
    assert "test/mae_non_min_mean" in logged_keys
    assert "test/mae_non_min_std" in logged_keys
    assert "test/mae_non_min_var" in logged_keys

    # Check file output
    log_file = tmp_path / "regression_metrics.txt"
    assert log_file.exists()
    content = log_file.read_text()
    assert "=== Overall Regression Metrics (Test Epoch) ===" in content
    assert "MSE Mean:" in content
    assert "=== Regression Metrics for Targets > 1.2000 (Test Epoch) ===" in content


def test_confusion_matrix_callback_test_stage_no_logger():
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 0
    trainer.loggers = []
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act & Assert
    callback.on_test_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, pl_module)


def test_confusion_matrix_callback_test_stage_with_wandb():
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 0
    
    # Mock WandbLogger and its experiment property
    wandb_logger = MagicMock(spec=WandbLogger)
    mock_experiment = MagicMock()
    type(wandb_logger).experiment = property(lambda self: mock_experiment)
    
    trainer.loggers = [wandb_logger]
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act
    callback.on_test_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, pl_module)
    
    # Assert
    mock_experiment.log.assert_called_once()
    logged_dict = mock_experiment.log.call_args[0][0]
    
    assert "test/confusion_matrix" in logged_dict
    assert "test/confusion_matrix_table" in logged_dict
    
    table = logged_dict["test/confusion_matrix_table"]
    assert table.columns == ["Actual \\ Predicted", "P0", "P1", "P2"]
    assert table.data[0] == ["Class 0", 1, 0, 0]
    assert table.data[1] == ["Class 1", 0, 1, 1]
    assert table.data[2] == ["Class 2", 1, 0, 0]


def test_confusion_matrix_callback_test_stage_logs_to_file(tmp_path):
    # Arrange
    callback = ConfusionMatrixCallback(frequency=1)
    
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 2
    trainer.loggers = []
    trainer.log_dir = str(tmp_path)
    
    pl_module = MagicMock(spec=LightningModule)
    pl_module.num_classes = 3
    
    outputs = {
        "preds": torch.tensor([0, 1, 2, 0]),
        "targets": torch.tensor([0, 1, 1, 2])
    }
    
    # Act
    callback.on_test_batch_end(trainer, pl_module, outputs, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer, pl_module)
    
    # Assert
    log_file = tmp_path / "confusion_matrix.txt"
    assert log_file.exists()
    
    content = log_file.read_text()
    assert "=== Confusion Matrix (Test) ===" in content
    assert "Actual \\ Predicted" in content
    assert "Class 0" in content
    assert "Class 1" in content
    assert "Class 2" in content




