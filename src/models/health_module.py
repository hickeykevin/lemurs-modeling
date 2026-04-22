from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
from functools import partial

class HealthLitModule(LightningModule):
    """
    LightningModule for Health (LSTM) classification task.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics for health classification (0-4 scores)
        self.train_acc = partial(Accuracy, task="multiclass")
        self.val_acc = partial(Accuracy, task="multiclass")
        self.test_acc = partial(Accuracy, task="multiclass")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:

        val_dataloader = self.trainer.datamodule.val_dataloader()
        y_list = []
        for batch in val_dataloader:
            _, y = batch
            y_list.append(y.cpu().numpy())
        
        y_train = np.concatenate(y_list, axis=0)
        self.num_classes = int(np.max(y_train)) + 1

        self.train_acc = self.train_acc(num_classes=self.num_classes)
        self.val_acc = self.val_acc(num_classes=self.num_classes)
        self.test_acc = self.test_acc(num_classes=self.num_classes)
        
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}




from flaml import AutoML

class FLAMLHealthModule(LightningModule):
    """A LightningModule wrapper for FLAML AutoML.
    
    This module fits a classic ML model (XGBoost, LGBM, etc.) during the 
    on_train_start hook and provides dummy steps to satisfy the Lightning trainer.
    """
    
    def __init__(self, automl_config: Dict[str, Any], task: str = "classification"):
        super().__init__()
        self.save_hyperparameters()
        
        self.automl = AutoML()
        self.task = task
        
        # Metrics for logging
        # We assume 6 classes based on previous context, but could be dynamic
        self.train_acc = partial(Accuracy, task="multiclass")
        self.val_acc = partial(Accuracy, task="multiclass")
        
    def _flatten_batch(self, x: torch.Tensor) -> np.ndarray:
        """Flattens [Batch, Time, Features] into [Batch, Time*Features] for tabular ML."""
        # x: [B, T, F] -> [B, T*F]
        return x.cpu().numpy().reshape(x.shape[0], -1)

    def on_train_start(self) -> None:
        """Collects the entire training set and fits the AutoML model."""
        self.print("--- FLAML AutoML: Starting Optimization ---")
        
        # Get the training dataloader from the datamodule
        train_dataloader = self.trainer.datamodule.train_dataloader()
        
        X_list, y_list = [], []
        for batch in train_dataloader:
            x, y = batch
            X_list.append(self._flatten_batch(x))
            y_list.append(y.cpu().numpy())
            
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        # count number of classes in the data
        num_classes = int(np.max(y_train)) + 1
        self.train_acc = self.train_acc(num_classes=num_classes)
        self.val_acc = self.val_acc(num_classes=num_classes)
        
        # Fit FLAML
        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            task=self.task,
            **self.hparams.automl_config
        )
        self.print(f"--- FLAML AutoML: Fit Complete. Best Model: {self.automl.best_estimator} ---")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy training step. Optimization happens in on_train_start."""
        # Return a zero loss with gradient required to keep Lightning happy
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluates the fitted FLAML model on the validation batch."""
        x, y = batch
        X_val = self._flatten_batch(x)
        
        # Predict
        y_pred = self.automl.predict(X_val)
        y_pred_tensor = torch.tensor(y_pred, device=self.device)
        
        # Update and log metrics
        self.val_acc(y_pred_tensor, y)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluates the fitted FLAML model on the test batch."""
        x, y = batch
        X_test = self._flatten_batch(x)
        
        y_pred = self.automl.predict(X_test)
        y_pred_tensor = torch.tensor(y_pred, device=self.device)
        
        # Log test metrics
        self.val_acc(y_pred_tensor, y) # Reuse val_acc for simplicity
        self.log("test/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        """Returns a dummy optimizer since we don't use backprop."""
        return torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.0)

if __name__ == "__main__":
    _ = HealthLitModule(None, None, None, None)
