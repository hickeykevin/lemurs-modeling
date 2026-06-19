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

