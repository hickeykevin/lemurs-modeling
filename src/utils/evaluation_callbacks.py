from typing import Any, Dict, List, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection, MaxMetric, MinMetric
from torchmetrics.classification import AUROC, F1Score, MulticlassConfusionMatrix, BinaryConfusionMatrix, Precision, Recall
from rich.table import Table
from rich.console import Console
from rich import box

class ConfusionMatrixCallback(Callback):
    """A Lightning callback that prints a confusion matrix at the end of each validation epoch.
    
    This callback collects predictions and targets from the validation step and 
    uses the Rich library to display a formatted confusion matrix in the terminal.

    Args:
        frequency (int): How often to print the confusion matrix (e.g., 5 means every 5 epochs).
    """

    def __init__(self, frequency: int = 1) -> None:
        """Initializes the ConfusionMatrixCallback.

        Args:
            frequency (int): How often to print the confusion matrix (e.g., 5 means every 5 epochs).
                Defaults to 1 (every epoch).
        """
        super().__init__()
        self.frequency = frequency
        self.preds: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.console = Console()

    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Collects predictions and targets from the validation batch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being validated.
            outputs (Optional[Dict[str, torch.Tensor]]): The outputs from the validation_step.
            batch (Any): The current batch of data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.
        """
        # Only collect data if we are going to print at the end of this epoch
        if trainer.current_epoch % self.frequency != 0:
            return

        if outputs is not None and "preds" in outputs and "targets" in outputs:
            self.preds.append(outputs["preds"].detach().cpu())
            self.targets.append(outputs["targets"].detach().cpu())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and prints the confusion matrix at the end of the validation epoch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being validated.
        """
        if trainer.current_epoch % self.frequency != 0:
            return

        if not self.preds:
            return

        # Concatenate all predictions and targets from the epoch
        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)
        
        # Determine the number of classes
        num_classes = getattr(pl_module, "num_classes", None)
        if num_classes is None:
            # Fallback if num_classes is not explicitly set on the module
            num_classes = int(torch.max(torch.max(all_preds), torch.max(all_targets)).item()) + 1

        # Compute confusion matrix
        if num_classes <= 2:
            cm_metric = BinaryConfusionMatrix()
            num_classes = 2
        else:
            cm_metric = MulticlassConfusionMatrix(num_classes=num_classes)
        cm = cm_metric(all_preds, all_targets)
        cm_np = cm.numpy().astype(int)

        # Create a Rich table for display
        table = Table(
            title=f"Confusion Matrix (Validation Epoch {trainer.current_epoch})",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Actual \\ Predicted", justify="right", style="cyan")
        for i in range(num_classes):
            table.add_column(f"P{i}", justify="center")

        # Add rows (Actual class i)
        for i in range(num_classes):
            row = [f"Class {i}"] + [str(cm_np[i, j]) for j in range(num_classes)]
            table.add_row(*row)

        # Capture the Rich table with color/style for terminal printing
        with self.console.capture() as capture:
            self.console.print(table)
        table_text = capture.get()

        # Print to terminal safely if tqdm is present, otherwise standard print
        try:
            from tqdm import tqdm
            tqdm.write(table_text)
        except ImportError:
            self.console.print("\n")
            self.console.print(table)
            self.console.print("\n")

        # Save confusion matrix to a file in the log directory
        import os
        log_dir = None
        if hasattr(trainer, "log_dir") and isinstance(trainer.log_dir, str):
            log_dir = trainer.log_dir
        elif hasattr(trainer, "default_root_dir") and isinstance(trainer.default_root_dir, str):
            log_dir = trainer.default_root_dir

        if log_dir is not None:
            # Capture a clean plain text table without ANSI colors/styles for file logging
            clean_console = Console(no_color=True, force_terminal=False)
            with clean_console.capture() as capture_clean:
                clean_console.print(table)
            clean_table_text = capture_clean.get()

            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, "confusion_matrix.txt")
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== Confusion Matrix (Validation Epoch {trainer.current_epoch}) ===\n")
                f.write(clean_table_text)
                f.write("\n")

        # Log to Wandb if WandbLogger is used
        from importlib.util import find_spec
        if find_spec("wandb"):
            import wandb
            from lightning.pytorch.loggers.wandb import WandbLogger

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    class_names = [f"Class {i}" for i in range(num_classes)]
                    
                    # Log as an interactive confusion matrix plot
                    cm_plot = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_targets.cpu().numpy(),
                        preds=all_preds.cpu().numpy(),
                        class_names=class_names,
                        title=f"Confusion Matrix (Val Epoch {trainer.current_epoch})"
                    )
                    
                    # Log as a wandb Table (matrix grid representation)
                    columns = ["Actual \\ Predicted"] + [f"P{i}" for i in range(num_classes)]
                    table_data = []
                    for i in range(num_classes):
                        row = [f"Class {i}"] + [int(cm_np[i, j]) for j in range(num_classes)]
                        table_data.append(row)
                    wandb_table = wandb.Table(data=table_data, columns=columns)
                    
                    logger.experiment.log({
                        "val/confusion_matrix": cm_plot,
                        "val/confusion_matrix_table": wandb_table
                    })
        
        # Reset collections for the next validation epoch
        self.preds = []
        self.targets = []


class ClassificationMetricsCallback(Callback):
    """A Lightning callback that tracks additional classification metrics (F1, AUROC).
    
    This callback uses a torchmetrics.MetricCollection to manage multiple metrics 
    efficiently. It updates metrics at each batch and logs them via the LightningModule 
    at the end of validation and test epochs.
    
    Structure allows for easy addition of new metrics to the collection.
    """

    def __init__(
        self, 
        f1_average: str = "macro", 
        auroc_average: str = "macro",
        precision_average: str = "macro",
        recall_average: str = "macro",
    ) -> None:
        """Initializes the ClassificationMetricsCallback.

        Args:
            f1_average (str): Averaging strategy for F1-Score (e.g., 'macro', 'micro', 'weighted').
                Defaults to "macro".
            auroc_average (str): Averaging strategy for AUROC (e.g., 'macro', 'weighted').
                Defaults to "macro".
            precision_average (str): Averaging strategy for Precision (e.g., 'macro', 'micro', 'weighted').
                Defaults to "macro".
            recall_average (str): Averaging strategy for Recall (e.g., 'macro', 'micro', 'weighted').
                Defaults to "macro".
        """
        super().__init__()
        self.f1_params = {"task": "multiclass", "average": f1_average}
        self.auroc_params = {"task": "multiclass", "average": auroc_average}
        self.precision_params = {"task": "multiclass", "average": precision_average}
        self.recall_params = {"task": "multiclass", "average": recall_average}
        
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        
        # Track the best metrics over validation epochs
        self.val_f1_best = MaxMetric()
        self.val_auroc_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()

    def _init_metrics(self, num_classes: int, device: torch.device) -> MetricCollection:
        """Initializes the MetricCollection with F1, AUROC, Precision, and Recall.

        To add new metrics, simply add them to this dictionary.

        Args:
            num_classes (int): The number of target classes.
            device (torch.device): The device (CPU/GPU) where metrics should be allocated.

        Returns:
            MetricCollection: A collection of initialized torchmetrics.
        """
        num_classes = max(2, num_classes)
        metrics = MetricCollection({
            "f1": F1Score(num_classes=num_classes, **self.f1_params),
            "auroc": AUROC(num_classes=num_classes, **self.auroc_params),
            "precision": Precision(num_classes=num_classes, **self.precision_params),
            "recall": Recall(num_classes=num_classes, **self.recall_params),
        })
        return metrics.to(device)


    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Resets metrics at the beginning of the validation epoch.

        This ensures that metrics are calculated independently for each epoch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being validated.
        """
        if trainer.sanity_checking:
            return
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[Dict[str, torch.Tensor]], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Updates validation metrics with results from the current batch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being validated.
            outputs (Optional[Dict[str, torch.Tensor]]): The outputs from the validation_step.
            batch (Any): The current batch of data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.
        """
        if trainer.sanity_checking:
            return
        if self.val_metrics is None:
            self.val_metrics = self._init_metrics(pl_module.num_classes, pl_module.device)
            
        if outputs is not None and "logits" in outputs and "targets" in outputs:
            self.val_metrics.update(outputs["logits"], outputs["targets"])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and logs validation metrics at the end of the epoch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being validated.
        """
        if trainer.sanity_checking:
            return
        if self.val_metrics is not None:
            output = self.val_metrics.compute()
            # Log all metrics in the collection
            for name, value in output.items():
                pl_module.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
                
                # Track the best metrics over validation epochs
                if name == "f1":
                    if self.val_f1_best.device != pl_module.device:
                        self.val_f1_best = self.val_f1_best.to(pl_module.device)
                    self.val_f1_best(value)
                    pl_module.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
                elif name == "auroc":
                    if self.val_auroc_best.device != pl_module.device:
                        self.val_auroc_best = self.val_auroc_best.to(pl_module.device)
                    self.val_auroc_best(value)
                    pl_module.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)
                elif name == "precision":
                    if self.val_precision_best.device != pl_module.device:
                        self.val_precision_best = self.val_precision_best.to(pl_module.device)
                    self.val_precision_best(value)
                    pl_module.log("val/precision_best", self.val_precision_best.compute(), sync_dist=True, prog_bar=True)
                elif name == "recall":
                    if self.val_recall_best.device != pl_module.device:
                        self.val_recall_best = self.val_recall_best.to(pl_module.device)
                    self.val_recall_best(value)
                    pl_module.log("val/recall_best", self.val_recall_best.compute(), sync_dist=True, prog_bar=True)
                    
            self.val_metrics.reset()

    def on_test_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Updates test metrics with results from the current batch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being tested.
            outputs (Optional[Dict[str, torch.Tensor]]): The outputs from the test_step.
            batch (Any): The current batch of data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.
        """
        if self.test_metrics is None:
            self.test_metrics = self._init_metrics(pl_module.num_classes, pl_module.device)
            
        if outputs is not None and "logits" in outputs and "targets" in outputs:
            self.test_metrics.update(outputs["logits"], outputs["targets"])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and logs test metrics at the end of the epoch.

        Args:
            trainer (Trainer): The Lightning trainer object.
            pl_module (LightningModule): The Lightning module being tested.
        """
        if self.test_metrics is not None:
            output = self.test_metrics.compute()
            # Log all metrics in the collection
            for name, value in output.items():
                pl_module.log(f"test/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
            self.test_metrics.reset()


class RegressionMetricsCallback(Callback):
    """A Lightning callback that tracks additional regression metrics (MSE, MAE).
    
    This callback updates metrics at each batch and logs them via the LightningModule 
    at the end of validation and test epochs.
    """

    def __init__(self) -> None:
        """Initializes the RegressionMetricsCallback."""
        super().__init__()
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        
        # Track the best (lowest) validation MSE over epochs
        self.val_mse_best = MinMetric()

    def _init_metrics(self, device: torch.device) -> MetricCollection:
        """Initializes the MetricCollection with MSE and MAE.

        Args:
            device (torch.device): The device (CPU/GPU) where metrics should be allocated.

        Returns:
            MetricCollection: A collection of initialized torchmetrics.
        """
        from torchmetrics import MeanSquaredError, MeanAbsoluteError
        metrics = MetricCollection({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
        })
        return metrics.to(device)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Resets metrics at the beginning of the validation epoch."""
        if trainer.sanity_checking:
            return
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Updates validation metrics with results from the current batch."""
        if trainer.sanity_checking:
            return
        if self.val_metrics is None:
            self.val_metrics = self._init_metrics(pl_module.device)
            
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            preds = outputs["preds"].squeeze()
            targets = outputs["targets"].squeeze()
            
            # Ensure tensors have at least 1 dimension
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
                
            self.val_metrics.update(preds, targets)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and logs validation metrics at the end of the epoch."""
        if trainer.sanity_checking:
            return
        if self.val_metrics is not None:
            output = self.val_metrics.compute()
            # Log all metrics in the collection
            for name, value in output.items():
                pl_module.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
                
                # Track the best metrics over validation epochs
                if name == "mse":
                    if self.val_mse_best.device != pl_module.device:
                        self.val_mse_best = self.val_mse_best.to(pl_module.device)
                    self.val_mse_best(value)
                    pl_module.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)
                    
            self.val_metrics.reset()

    def on_test_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Updates test metrics with results from the current batch."""
        if self.test_metrics is None:
            self.test_metrics = self._init_metrics(pl_module.device)
            
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            preds = outputs["preds"].squeeze()
            targets = outputs["targets"].squeeze()
            
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
                
            self.test_metrics.update(preds, targets)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and logs test metrics at the end of the epoch."""
        if self.test_metrics is not None:
            output = self.test_metrics.compute()
            # Log all metrics in the collection
            for name, value in output.items():
                pl_module.log(f"test/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
            self.test_metrics.reset()


class TargetDistributionCallback(Callback):
    """A Lightning callback that logs the distribution of targets/answers to WandB.
    
    Logs a histogram of the target values exactly once during the first validation run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.targets: List[torch.Tensor] = []
        self.has_logged = False

    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        if trainer.sanity_checking or self.has_logged:
            return

        if outputs is not None and "targets" in outputs:
            self.targets.append(outputs["targets"].detach().cpu())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or self.has_logged:
            return

        if not self.targets:
            return

        # Concatenate all targets
        all_targets = torch.cat(self.targets).numpy().flatten()
        
        # Log to Wandb if WandbLogger is used
        from importlib.util import find_spec
        if find_spec("wandb"):
            import wandb
            from lightning.pytorch.loggers.wandb import WandbLogger

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    import matplotlib.pyplot as plt
                    import io
                    from PIL import Image
                    
                    # Create the matplotlib histogram
                    plt.figure(figsize=(8, 6))
                    plt.hist(all_targets, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
                    plt.title("Validation Target Distribution", fontsize=14)
                    plt.xlabel("Target Value", fontsize=12)
                    plt.ylabel("Count", fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    
                    # Save plot to an in-memory buffer and convert to PIL Image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    img = Image.open(buf)
                    plt.close()
                    
                    logger.experiment.log({
                        "val/target_distribution": wandb.Image(img, caption="Validation Target Distribution"),
                    })
                    self.has_logged = True
        
        # Reset targets list
        self.targets = []



