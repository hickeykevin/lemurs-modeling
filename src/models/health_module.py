from typing import Any, Dict, Tuple, Optional, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric, MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
from functools import partial

class HealthLitModule(LightningModule):
    """A LightningModule for Health (LSTM) classification tasks.

    This module handles the training, validation, and testing logic for a sequence-based
    LSTM model designed to predict health outcomes from multimodal data.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation Loop (validation_step)
        - Test Loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        class_weights: Optional[List[float]] = None,
        user_id_dropout: float = 0.0,
    ) -> None:
        """Initializes the HealthLitModule.

        Args:
            net (torch.nn.Module): The neural network model (e.g., SimpleLSTM).
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
            compile (bool): Whether to compile the model for faster execution.
            class_weights (Optional[List[float]]): Optional weights for each class in the loss function.
            user_id_dropout (float): Probability of dropping out user index (replacing with 0) during training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures hyperparameters are stored in the checkpoint
        self.save_hyperparameters(logger=False)

        self.net = net
        
        # Initialize criterion with optional class weights
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float))
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics for health classification (0-4 scores)
        # We use partial initialization because num_classes is determined at runtime
        self.train_acc = partial(Accuracy, task="multiclass")
        self.val_acc = partial(Accuracy, task="multiclass")
        self.test_acc = partial(Accuracy, task="multiclass")

        # Averaging metrics over the entire epoch
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Track the best validation accuracy across epochs
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, features].
            user_idx: Optional user indices tensor of shape [batch_size].

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        return self.net(x, user_idx)

    def on_train_start(self) -> None:
        pass


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single step through the model.

        Args:
            batch: A tuple containing (features, targets, user_indices).

        Returns:
            A tuple of (loss, predictions, targets, logits).
        """
        x, y, user_idx = batch
        
        # Apply user ID dropout during training to fallback to the population average (index 0)
        if self.training and self.hparams.user_id_dropout > 0:
            mask = torch.rand(user_idx.shape, device=user_idx.device) < self.hparams.user_id_dropout
            user_idx = torch.where(mask, torch.zeros_like(user_idx), user_idx)

        logits = self.forward(x, user_idx)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A tuple containing (features, targets).
            batch_idx: The index of the current batch.

        Returns:
            The loss tensor.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A tuple containing (features, targets).
            batch_idx: The index of the current batch.
        """
        loss, preds, targets, logits = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of the validation epoch.

        Used to update the best validation accuracy.
        """
        acc = self.val_acc.compute()  # Get accuracy from the current epoch
        self.val_acc_best(acc)        # Update max metric
        
        # Log best accuracy so far
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A tuple containing (features, targets).
            batch_idx: The index of the current batch.
        """
        loss, preds, targets, logits = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def setup(self, stage: str, **kwargs) -> None:
        """Lightning hook for setting up data or model before training.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        if not hasattr(self, "num_classes"):
            num_classes = kwargs.get("num_classes")
            if num_classes is not None:
                self.num_classes = num_classes
            elif self._trainer is not None and self.trainer.datamodule is not None:
                # We look at the validation set to find the range of labels
                val_dataloader = self.trainer.datamodule.val_dataloader()
                y_list = []
                for batch in val_dataloader:
                    _, y = batch[:2]
                    y_list.append(y.cpu().numpy())
                
                y_train = np.concatenate(y_list, axis=0)
                self.num_classes = int(np.max(y_train)) + 1
            else:
                raise RuntimeError("Cannot determine num_classes: neither num_classes arg nor trainer is available.")
        # Dynamically build user embedding on the net if user_to_idx is present on the datamodule
        if self._trainer is not None and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "user_to_idx") and dm.user_to_idx is not None:
                self.num_users = len(dm.user_to_idx) + 1
                if hasattr(self.net, "init_user_embedding"):
                    self.net.init_user_embedding(self.num_users)
                    self.net.to(self.device)  # Make sure embedding weights are moved to correct device

        # Instantiate the actual Accuracy metrics with the correct task type
        if self.num_classes <= 2:
            # For binary tasks (0 or 1), torchmetrics Accuracy doesn't use num_classes
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
        else:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        # Reset metrics to ensure a clean start
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Returns:
            A dictionary containing the optimizer and optionally a learning rate scheduler.
        """
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
    
    This module fits a classic tabular ML model (e.g., XGBoost, LGBM, CatBoost) 
    during the `on_train_start` hook. It bypasses traditional backpropagation
    by using a dummy optimizer and returning zero loss.

    This is useful for comparing deep learning models (like LSTMs) against
    automated machine learning benchmarks on the same data pipeline.
    """
    
    def __init__(self, automl_config: Dict[str, Any], task: str = "classification"):
        """Initializes the FLAMLHealthModule.

        Args:
            automl_config (Dict[str, Any]): Configuration dictionary for FLAML's fit method.
            task (str): Type of AutoML task (e.g., 'classification', 'regression').
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.automl = AutoML()
        self.task = task
        
        # Metrics for logging
        self.train_acc = partial(Accuracy, task="multiclass")
        self.val_acc = partial(Accuracy, task="multiclass")
        self.test_acc = partial(Accuracy, task="multiclass")
        
    def _flatten_batch(self, x: torch.Tensor) -> np.ndarray:
        """Flattens sequential data for tabular ML models.

        Args:
            x: Input tensor of shape [batch_size, time_steps, features].

        Returns:
            NumPy array of shape [batch_size, time_steps * features].
        """
        # x: [B, T, F] -> [B, T*F]
        return x.cpu().numpy().reshape(x.shape[0], -1)

    def setup(self, stage: str, **kwargs) -> None:
        """Collects the entire training set and fits the AutoML model."""
        if not hasattr(self, "num_classes"):
            num_classes = kwargs.get("num_classes")
            if num_classes is not None:
                self.num_classes = num_classes
            elif hasattr(self, "trainer") and self.trainer is not None:
                # Determine classes by looking at validation labels
                val_dataloader = self.trainer.datamodule.val_dataloader()
                y_list = []
                for batch in val_dataloader:
                    _, y = batch[:2]
                    y_list.append(y.cpu().numpy())
                y_val = np.concatenate(y_list, axis=0)
                self.num_classes = int(np.max(y_val)) + 1
            else:
                self.num_classes = 2  # Default binary fallback
                
            # Instantiate correct accuracy metrics (binary or multiclass)
            if self.num_classes <= 2:
                self.train_acc = Accuracy(task="binary")
                self.val_acc = Accuracy(task="binary")
                self.test_acc = Accuracy(task="binary")
            else:
                self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        if stage == "fit" and not hasattr(self.automl, "best_estimator"):
            self.print("--- FLAML AutoML: Starting Optimization ---")
            
            # Get the training dataloader from the datamodule
            train_dataloader = self.trainer.datamodule.train_dataloader()
            
            X_list, y_list = [], []
            for batch in train_dataloader:
                x, y = batch[:2]
                X_list.append(self._flatten_batch(x))
                y_list.append(y.cpu().numpy())
                
            X_train = np.concatenate(X_list, axis=0)
            y_train = np.concatenate(y_list, axis=0)

            # Double check class bounds from actual training data
            num_classes_train = int(np.max(y_train)) + 1
            if num_classes_train > self.num_classes:
                self.num_classes = num_classes_train
                if self.num_classes <= 2:
                    self.train_acc = Accuracy(task="binary")
                    self.val_acc = Accuracy(task="binary")
                    self.test_acc = Accuracy(task="binary")
                else:
                    self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                    self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
                    self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            
            # Fit FLAML (this may take some time depending on automl_config.time_budget)
            self.automl.fit(
                X_train=X_train,
                y_train=y_train,
                task=self.task,
                **self.hparams.automl_config
            )
            self.print(f"--- FLAML AutoML: Fit Complete. Best Model: {self.automl.best_estimator} ---")


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy training step. 
        
        Optimization happens in setup, so we stop training after the first batch.
        """
        self.trainer.should_stop = True
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Evaluates the fitted FLAML model on a validation batch.

        Args:
            batch: A tuple containing (features, targets, ...).
            batch_idx: The index of the current batch.
        """
        x, y = batch[:2]
        X_val = self._flatten_batch(x)
        
        # Predict using the fitted AutoML object
        y_pred = self.automl.predict(X_val)
        y_pred_tensor = torch.tensor(y_pred, device=self.device)
        
        # Try to get probabilities for AUROC
        try:
            y_prob = self.automl.predict_proba(X_val)
            logits = torch.tensor(y_prob, device=self.device)
        except:
            # Fallback if the estimator doesn't support predict_proba
            logits = torch.nn.functional.one_hot(y_pred_tensor, num_classes=self.num_classes).float()

        # Update and log metrics
        self.val_acc(y_pred_tensor, y)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"preds": y_pred_tensor, "targets": y, "logits": logits}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Evaluates the fitted FLAML model on a test batch.

        Args:
            batch: A tuple containing (features, targets, ...).
            batch_idx: The index of the current batch.
        """
        x, y = batch[:2]
        X_test = self._flatten_batch(x)
        
        y_pred = self.automl.predict(X_test)
        y_pred_tensor = torch.tensor(y_pred, device=self.device)

        # Try to get probabilities for AUROC
        try:
            y_prob = self.automl.predict_proba(X_test)
            logits = torch.tensor(y_prob, device=self.device)
        except:
            # Fallback
            logits = torch.nn.functional.one_hot(y_pred_tensor, num_classes=self.num_classes).float()
        
        # Log test metrics
        self.test_acc(y_pred_tensor, y)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"preds": y_pred_tensor, "targets": y, "logits": logits}

    def configure_optimizers(self) -> Any:
        """Returns a dummy optimizer.
        
        Tabular AutoML models are not trained via backpropagation.
        """
        return torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.0)


class BaselineHealthModule(LightningModule):
    """A LightningModule for baseline health classification tasks.
    
    Supported Strategies:
        - "lag": Predicts the current label using the label from the user's previous 
          survey (requires x to contain the previous label index).
        - "most_frequent": Always predicts the most frequent class observed in training.
    """
    
    def __init__(self, strategy: str = "lag", num_classes: Optional[int] = None):
        """Initializes the BaselineHealthModule.
        
        Args:
            strategy (str): Baseline strategy to use ("lag" or "most_frequent").
            num_classes (Optional[int]): Number of target classes.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.strategy = strategy
        self.majority_class = None
        
        # Metrics for logging (mirrors HealthLitModule)
        self.val_acc = partial(Accuracy, task="multiclass")
        self.test_acc = partial(Accuracy, task="multiclass")
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        if num_classes:
            self.num_classes = num_classes
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def setup(self, stage: str, **kwargs) -> None:
        """Determines num_classes and majority class if needed."""
        # Get labels from training set to find majority class
        dm = self.trainer.datamodule
        train_dataloader = dm.train_dataloader()
        
        y_list = []
        for batch in train_dataloader:
            _, y = batch[:2]
            y_list.append(y.cpu().numpy())
        y_train = np.concatenate(y_list, axis=0)

        if not hasattr(self, "num_classes"):
            self.num_classes = int(np.max(y_train)) + 1
            self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        if self.strategy == "most_frequent":
            # Calculate the mode of the training labels
            counts = np.bincount(y_train)
            self.majority_class = int(np.argmax(counts))
            self.print(f"[Baseline] Strategy: most_frequent | Majority Class: {self.majority_class}")
        else:
            self.print(f"[Baseline] Strategy: lag")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produces logits based on the selected strategy."""
        batch_size = x.shape[0]
        logits = torch.full((batch_size, self.num_classes), -100.0, device=self.device)
        
        if self.strategy == "most_frequent":
            logits[:, self.majority_class] = 100.0
        elif self.strategy == "lag":
            # Extract the previous label from x [B, 1, 1]
            prev_labels = x[:, 0, 0].long()
            for i in range(batch_size):
                label = prev_labels[i].item()
                if 0 <= label < self.num_classes:
                    logits[i, label] = 100.0
                else:
                    # Fallback to majority class if history is missing
                    if self.majority_class is not None:
                        logits[i, self.majority_class] = 100.0
                    else:
                        logits[i, :] = 0.0 # Uniform
        
        return logits

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch[:2]
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.trainer.should_stop = True
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets, logits = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets, logits = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.0)

if __name__ == "__main__":
    _ = HealthLitModule(None, None, None, None)


class HealthRegressionLitModule(LightningModule):
    """A LightningModule for Health (LSTM) regression tasks.

    This module handles the training, validation, and testing logic for a sequence-based
    LSTM model designed to predict continuous outcomes from multimodal data.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        user_id_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.MSELoss()

        # Averaging metrics over the entire epoch
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x, user_idx)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, user_idx = batch
        
        # Apply user ID dropout during training to fallback to the population average (index 0)
        if self.training and self.hparams.user_id_dropout > 0:
            mask = torch.rand(user_idx.shape, device=user_idx.device) < self.hparams.user_id_dropout
            user_idx = torch.where(mask, torch.zeros_like(user_idx), user_idx)

        logits = self.forward(x, user_idx) # shape [Batch, output_size=1]
        preds = logits.squeeze(-1)
        loss = self.criterion(preds, y.float())
        return loss, preds, y, logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets, logits = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Yield preds and targets so validation callbacks can compute regression metrics
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets, logits = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Yield preds and targets so test callbacks can compute regression metrics
        return {"loss": loss, "preds": preds, "targets": targets}

    def setup(self, stage: str, **kwargs) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        # Dynamically build user embedding on the net if user_to_idx is present on the datamodule
        if self._trainer is not None and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "user_to_idx") and dm.user_to_idx is not None:
                self.num_users = len(dm.user_to_idx) + 1
                if hasattr(self.net, "init_user_embedding"):
                    self.net.init_user_embedding(self.num_users)
                    self.net.to(self.device)  # Make sure embedding weights are moved to correct device

        self.val_loss.reset()

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

