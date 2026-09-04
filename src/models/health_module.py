from typing import Any, Dict, Tuple, Optional, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric, MeanSquaredError, MeanAbsoluteError
import numpy as np
import pickle
from functools import partial


def _resolve_num_classes(trainer: Any, explicit: Optional[int], scan_fallback_labels) -> int:
    """Determines the classification head width for the current fold.

    Priority order:
      1. An explicit ``num_classes`` (passed to ``setup()``, or set at construction).
      2. The aggregator's declared ``num_classes`` — the same source that already
         sizes the network's output layer via ``net.output_size`` in config, so
         this is the only choice that cannot drift out of sync with the model.
      3. Scanning a dataloader's realized labels, as a last resort.

    (3) is unsafe under repeated grouped CV. The outer test fold is stratified on
    whether a user is ever positive, but the inner validation carve-out is
    re-stratified from an already-shrunken training pool and can fall back to
    unstratified grouping if the minority stratum runs too small (see
    ``CVHealthDataModule._grouped_split``). That means a fold's validation split
    can end up single-class purely by chance. Inferring ``num_classes`` from that
    split would produce a class-weight tensor sized for one class while the
    network's actual output layer — sized from the aggregator — has two,
    breaking ``CrossEntropyLoss`` for that fold only. Preferring the aggregator's
    declared count avoids ever taking that path in practice.
    """
    if explicit is not None:
        return int(explicit)

    dm = getattr(trainer, "datamodule", None) if trainer is not None else None
    hparams = getattr(dm, "hparams", None) if dm is not None else None
    aggregator = getattr(hparams, "aggregator", None) if hparams is not None else None
    declared = getattr(aggregator, "num_classes", None)
    if declared is not None:
        return int(declared)

    import logging
    logging.getLogger(__name__).warning(
        "aggregator.num_classes unavailable; falling back to scanning a dataloader "
        "for the label range. Under repeated CV a fold's split can be single-class "
        "purely by chance, which would under-count classes here — set num_classes "
        "on the aggregator config if this warning appears routinely."
    )
    return scan_fallback_labels()


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
        auto_class_weights: bool = False,
        use_sequence_data: bool = True,
    ) -> None:
        """Initializes the HealthLitModule.

        Args:
            net (torch.nn.Module): The neural network model (e.g., SimpleLSTM).
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
            compile (bool): Whether to compile the model for faster execution.
            class_weights (Optional[List[float]]): Optional weights for each class in the loss function.
            use_sequence_data (bool): Whether to include sequence data.
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

        # Averaging metrics over the entire epoch
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, features].
            demographics: Optional demographics tensor of shape [batch_size, demographics_dim].

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        return self.net(x, demographics)

    def on_train_start(self) -> None:
        pass

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """Computes balanced (inverse-frequency) class weights from the training labels.

        Uses the sklearn "balanced" heuristic ``weight_c = n_samples / (n_classes * count_c)``.
        Runs against the current training split, so under cross-validation the weights
        are recomputed per-fold. Classes absent from the split are floored to a count of
        1 to avoid division by zero.

        Returns:
            A ``[num_classes]`` float tensor on the module device, or ``None`` if the
            training dataloader is unavailable.
        """
        if self._trainer is None or self.trainer.datamodule is None:
            return None
        train_loader = self.trainer.datamodule.train_dataloader()
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for batch in train_loader:
            y = batch[1].detach().cpu().numpy().astype(int)
            counts += np.bincount(y, minlength=self.num_classes)[: self.num_classes]
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float, device=self.device)

    def _maybe_strip_index(self, batch: Any, stage: str) -> Any:
        """Drops a trailing sample-index tensor before the fixed-length dispatch below runs.

        ``HealthDataset.__getitem__`` appends ``idx`` as the true last tuple
        element when built with ``return_index=True`` (see
        ``WalkForwardHealthDataModule``, which sets this on ``data_val``/
        ``data_test`` but not ``data_train``). Undetected, that idx tensor
        would be destructured by the length-based dispatch below as if it
        were the demographics tensor and fed straight into the network,
        silently corrupting both training and evaluation rather than merely
        confusing a prediction-collecting callback.

        Checked per call against the *current* stage's dataset (rather than
        a static flag on this module) so it can never drift out of sync with
        what the datamodule actually built for that stage, and so the same
        module instance handles a datamodule where only some stages return
        an index (as ``WalkForwardHealthDataModule`` does) correctly.

        Uses ``self._trainer`` (the private attribute), not the ``trainer``
        property: the property raises ``RuntimeError`` when this module is
        not attached to a ``Trainer`` (e.g. ``model_step`` called directly in
        a unit test on a bare module), rather than returning ``None`` the way
        a plain ``getattr`` fallback would expect.
        """
        dm = getattr(self._trainer, "datamodule", None) if self._trainer is not None else None
        dataset = getattr(dm, f"data_{stage}", None) if dm is not None else None
        if getattr(dataset, "return_index", False):
            return batch[:-1]
        return batch

    def model_step(
        self, batch: Any, stage: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single step through the model.

        Args:
            batch: A tuple containing features, targets, user_indices, etc.
            stage: One of "train", "val", "test" -- which dataset's
                ``return_index`` flag to check (see ``_maybe_strip_index``).

        Returns:
            A tuple of (loss, predictions, targets, logits).
        """
        batch = self._maybe_strip_index(batch, stage)

        if len(batch) == 4:
            x, y, _, demographics = batch
        elif len(batch) == 6:
            x, y, _, _, _, demographics = batch
        elif len(batch) == 5:
            x, y, _, _, _ = batch
            demographics = None
        else:
            x, y = batch[0], batch[1]
            demographics = None

        logits = self.forward(x, demographics)
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
        loss, preds, targets, _ = self.model_step(batch, stage="train")

        # Update metrics
        self.train_loss(loss)

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A tuple containing (features, targets).
            batch_idx: The index of the current batch.
        """
        loss, preds, targets, logits = self.model_step(batch, stage="val")

        # Update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}



    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A tuple containing (features, targets).
            batch_idx: The index of the current batch.
        """
        loss, preds, targets, logits = self.model_step(batch, stage="test")

        # Update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def setup(self, stage: str, **kwargs) -> None:
        """Lightning hook for setting up data or model before training.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        if not hasattr(self, "num_classes"):
            explicit = kwargs.get("num_classes")
            if explicit is None and (self._trainer is None or self.trainer.datamodule is None):
                raise RuntimeError("Cannot determine num_classes: neither num_classes arg nor trainer is available.")

            def _scan_val_labels() -> int:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                y_list = [batch[1].cpu().numpy() for batch in val_dataloader]
                return int(np.max(np.concatenate(y_list, axis=0))) + 1

            self.num_classes = _resolve_num_classes(self._trainer, explicit, _scan_val_labels)
        # Dynamically build input size, user embedding, and demographics support on the net
        if self._trainer is not None and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "data_train") and dm.data_train is not None:
                train_ds = dm.data_train
                if len(train_ds) > 0:
                    sample = train_ds[0]
                    input_size = sample[0].shape[-1]
                    if hasattr(self.net, "init_input_size"):
                        self.net.init_input_size(input_size)
            if hasattr(dm, "demographics_dim") and dm.demographics_dim is not None:
                if hasattr(self.net, "init_demographics"):
                    self.net.init_demographics(dm.demographics_dim)
            self.net.to(self.device)  # Make sure weights are moved to correct device

        # Auto-compute inverse-frequency class weights from the training split.
        # Skipped when explicit class_weights were provided. Recomputed each time
        # setup() runs (e.g. per fold under cross-validation).
        if getattr(self.hparams, "auto_class_weights", False) and self.hparams.class_weights is None:
            weights = self._compute_class_weights()
            if weights is not None:
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
                self.print(f"[AutoClassWeights] inverse-frequency weights: {[round(w, 3) for w in weights.tolist()]}")

        # Reset metrics to ensure a clean start
        self.val_loss.reset()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Dynamically adjust demographics shapes if present in state_dict before loading."""
        fc_weight_key = "net.fc.weight"
        if fc_weight_key in state_dict:
            checkpoint_in_features = state_dict[fc_weight_key].shape[1]
            current_in_features = self.net.fc.in_features
            if checkpoint_in_features > current_in_features:
                extra_dim = checkpoint_in_features - current_in_features
                if hasattr(self.net, "init_demographics"):
                    self.net.init_demographics(extra_dim)

        lstm_weight_key = "net.lstm.weight_ih_l0"
        if lstm_weight_key in state_dict:
            checkpoint_input_size = state_dict[lstm_weight_key].shape[1]
            if hasattr(self.net, "init_input_size"):
                self.net.init_input_size(checkpoint_input_size)

        return super().load_state_dict(state_dict, strict=strict)

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
    
    def __init__(
        self,
        automl_config: Dict[str, Any],
        task: str = "classification",
        auto_class_weights: bool = False,
        class_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """Initializes the FLAMLHealthModule.

        Args:
            automl_config (Dict[str, Any]): Configuration dictionary for FLAML's fit method.
            task (str): Type of AutoML task (e.g., 'classification', 'regression').
            auto_class_weights (bool): Whether to compute inverse-frequency sample weights automatically.
            class_weights (Optional[List[float]]): Explicit class weights list. Overrides auto_class_weights if set.
        """
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(automl_config, DictConfig):
                automl_config = OmegaConf.to_container(automl_config, resolve=True)
        except ImportError:
            pass

        super().__init__()
        self.save_hyperparameters()
        
        self.automl = AutoML()
        self.task = task

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
            explicit = kwargs.get("num_classes")
            if explicit is None and not (hasattr(self, "trainer") and self.trainer is not None):
                self.num_classes = 2  # Default binary fallback
            else:
                def _scan_val_labels() -> int:
                    val_dataloader = self.trainer.datamodule.val_dataloader()
                    y_list = [batch[1].cpu().numpy() for batch in val_dataloader]
                    return int(np.max(np.concatenate(y_list, axis=0))) + 1

                self.num_classes = _resolve_num_classes(
                    getattr(self, "trainer", None), explicit, _scan_val_labels
                )

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

            sample_weight = None
            explicit_weights = getattr(self.hparams, "class_weights", None)
            auto_weights = getattr(self.hparams, "auto_class_weights", False)

            if explicit_weights is not None:
                weights = np.array(explicit_weights, dtype=np.float64)
                sample_weight = weights[y_train.astype(int)]
                self.print(f"[FLAML ClassWeights] explicit weights: {[round(w, 3) for w in weights.tolist()]}")
            elif auto_weights:
                counts = np.bincount(y_train.astype(int), minlength=self.num_classes)[: self.num_classes].astype(np.float64)
                counts = np.maximum(counts, 1.0)
                weights = counts.sum() / (self.num_classes * counts)
                sample_weight = weights[y_train.astype(int)]
                self.print(f"[FLAML AutoClassWeights] inverse-frequency weights: {[round(w, 3) for w in weights.tolist()]}")

            fit_kwargs = dict(self.hparams.automl_config)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight

            # Fit FLAML (this may take some time depending on automl_config.time_budget)
            self.automl.fit(
                X_train=X_train,
                y_train=y_train,
                task=self.task,
                **fit_kwargs
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
        

        
        return {"preds": y_pred_tensor, "targets": y, "logits": logits}

    def configure_optimizers(self) -> Any:
        """Returns a dummy optimizer.
        
        Tabular AutoML models are not trained via backpropagation.
        """
        return torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.0)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves the fitted FLAML AutoML object and metadata into the checkpoint."""
        if hasattr(self.automl, "best_estimator") and self.automl.best_estimator is not None:
            checkpoint["flaml_automl"] = pickle.dumps(self.automl)
        if hasattr(self, "num_classes"):
            checkpoint["num_classes"] = self.num_classes

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restores the fitted FLAML AutoML object and metadata from the checkpoint."""
        if "flaml_automl" in checkpoint:
            self.automl = pickle.loads(checkpoint["flaml_automl"])
        if "num_classes" in checkpoint:
            self.num_classes = checkpoint["num_classes"]



class BaselineHealthModule(LightningModule):
    """A LightningModule for baseline health classification tasks.
    
    Supported Strategies:
        - "lag": Predicts the current label using the label from the user's previous 
          survey (requires x to contain the previous label index).
        - "most_frequent": Always predicts the most frequent class observed in training.
    """
    
    def __init__(self, strategy: str = "lag", num_classes: Optional[int] = None, **kwargs):
        """Initializes the BaselineHealthModule.
        
        Args:
            strategy (str): Baseline strategy to use ("lag" or "most_frequent").
            num_classes (Optional[int]): Number of target classes.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.strategy = strategy
        self.majority_class = None
        

        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        if num_classes:
            self.num_classes = num_classes

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
            self.num_classes = _resolve_num_classes(
                self._trainer, None, lambda: int(np.max(y_train)) + 1
            )

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
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets, logits = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.MSELoss()

        # Averaging metrics over the entire epoch
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x, demographics)

    def model_step(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 4:
            x, y, _, demographics = batch
        elif len(batch) == 6:
            x, y, _, _, _, demographics = batch
        elif len(batch) == 5:
            x, y, _, _, _ = batch
            demographics = None
        else:
            x, y = batch[0], batch[1]
            demographics = None

        logits = self.forward(x, demographics) # shape [Batch, output_size=1]
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

        # Dynamically build input size, user embedding, and demographics support on the net
        if self._trainer is not None and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "data_train") and dm.data_train is not None:
                train_ds = dm.data_train
                if len(train_ds) > 0:
                    sample = train_ds[0]
                    input_size = sample[0].shape[-1]
                    if hasattr(self.net, "init_input_size"):
                        self.net.init_input_size(input_size)
            if hasattr(dm, "demographics_dim") and dm.demographics_dim is not None:
                if hasattr(self.net, "init_demographics"):
                    self.net.init_demographics(dm.demographics_dim)
            self.net.to(self.device)  # Make sure weights are moved to correct device

        self.val_loss.reset()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Dynamically adjust demographics shapes if present in state_dict before loading."""
        fc_weight_key = "net.fc.weight"
        if fc_weight_key in state_dict:
            checkpoint_in_features = state_dict[fc_weight_key].shape[1]
            current_in_features = self.net.fc.in_features
            if checkpoint_in_features > current_in_features:
                extra_dim = checkpoint_in_features - current_in_features
                if hasattr(self.net, "init_demographics"):
                    self.net.init_demographics(extra_dim)

        lstm_weight_key = "net.lstm.weight_ih_l0"
        if lstm_weight_key in state_dict:
            checkpoint_input_size = state_dict[lstm_weight_key].shape[1]
            if hasattr(self.net, "init_input_size"):
                self.net.init_input_size(checkpoint_input_size)

        return super().load_state_dict(state_dict, strict=strict)

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

