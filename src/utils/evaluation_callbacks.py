from typing import Any, Dict, List, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection, MaxMetric, MinMetric
from torchmetrics.classification import AUROC, F1Score, MulticlassConfusionMatrix, BinaryConfusionMatrix, Precision, Recall
from torchmetrics.wrappers import BootStrapper
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

    def __init__(self, frequency: int = 1, print_report: bool = True) -> None:
        """Initializes the ConfusionMatrixCallback.

        Args:
            frequency (int): How often to print the confusion matrix (e.g., 5 means every 5 epochs).
                Defaults to 1 (every epoch).
            print_report (bool): Whether to print the scikit-learn classification report.
                Defaults to False.
        """
        super().__init__()
        self.frequency = frequency
        self.print_report = print_report
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

    def _print_and_log_confusion_matrix(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Computes, prints, and logs the confusion matrix for validation or test stage."""
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

        stage_title = "Validation" if stage == "val" else "Test"
        epoch_info = f" Epoch {trainer.current_epoch}" if stage == "val" else ""

        # Create a Rich table for display
        table = Table(
            title=f"Confusion Matrix ({stage_title}{epoch_info})",
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

        report_text = ""
        if self.print_report:
            from sklearn.metrics import classification_report
            y_true = all_targets.cpu().numpy()
            y_pred = all_preds.cpu().numpy()
            target_names = [f"Class {i}" for i in range(num_classes)]
            labels = list(range(num_classes))
            report = classification_report(
                y_true, 
                y_pred, 
                labels=labels, 
                target_names=target_names, 
                zero_division=0
            )
            report_text = f"\n=== Classification Report ({stage_title}{epoch_info}) ===\n{report}\n"

        # Capture the Rich table with color/style for terminal printing
        with self.console.capture() as capture:
            self.console.print(table)
        table_text = capture.get()

        # Print to terminal safely if tqdm is present, otherwise standard print
        try:
            from tqdm import tqdm
            tqdm.write(table_text)
            if self.print_report:
                tqdm.write(report_text)
        except ImportError:
            self.console.print("\n")
            self.console.print(table)
            self.console.print("\n")
            if self.print_report:
                self.console.print(report_text)

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
                f.write(f"\n=== Confusion Matrix ({stage_title}{epoch_info}) ===\n")
                f.write(clean_table_text)
                f.write("\n")
                if self.print_report:
                    f.write(report_text)

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
                        title=f"Confusion Matrix ({stage_title}{epoch_info})"
                    )
                    
                    # Log as a wandb Table (matrix grid representation)
                    columns = ["Actual \\ Predicted"] + [f"P{i}" for i in range(num_classes)]
                    table_data = []
                    for i in range(num_classes):
                        row = [f"Class {i}"] + [int(cm_np[i, j]) for j in range(num_classes)]
                        table_data.append(row)
                    wandb_table = wandb.Table(data=table_data, columns=columns)
                    
                    logger.experiment.log({
                        f"{stage}/confusion_matrix": cm_plot,
                        f"{stage}/confusion_matrix_table": wandb_table
                    })
        
        # Reset collections for the next validation epoch
        self.preds = []
        self.targets = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and prints the confusion matrix at the end of the validation epoch."""
        if trainer.current_epoch % self.frequency != 0:
            return
        self._print_and_log_confusion_matrix(trainer, pl_module, stage="val")

    def on_test_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict[str, torch.Tensor]], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Collects predictions and targets from the test batch."""
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            self.preds.append(outputs["preds"].detach().cpu())
            self.targets.append(outputs["targets"].detach().cpu())

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and prints the confusion matrix at the end of the test epoch."""
        self._print_and_log_confusion_matrix(trainer, pl_module, stage="test")


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
        num_bootstraps: int = 200,
        sampling_strategy: str = "poisson",
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
            num_bootstraps (int): The number of bootstrap resamples to use in the test stage.
                Defaults to 200.
            sampling_strategy (str): The sampling strategy to use for BootStrapper ('poisson' or 'multinomial').
                Defaults to "poisson".
        """
        super().__init__()
        self.f1_params = {"task": "multiclass", "average": f1_average}
        self.auroc_params = {"task": "multiclass", "average": auroc_average}
        self.precision_params = {"task": "multiclass", "average": precision_average}
        self.recall_params = {"task": "multiclass", "average": recall_average}
        self.num_bootstraps = num_bootstraps
        self.sampling_strategy = sampling_strategy
        
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        
        # Track the best metrics over validation epochs
        self.val_f1_best = MaxMetric()
        self.val_auroc_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()

    def _init_metrics(self, num_classes: int, device: torch.device, bootstrap: bool = False) -> MetricCollection:
        """Initializes the MetricCollection with F1, AUROC, Precision, and Recall.

        To add new metrics, simply add them to this dictionary.

        Args:
            num_classes (int): The number of target classes.
            device (torch.device): The device (CPU/GPU) where metrics should be allocated.
            bootstrap (bool): Whether to wrap metrics in a BootStrapper. Defaults to False.

        Returns:
            MetricCollection: A collection of initialized torchmetrics.
        """
        num_classes = max(2, num_classes)
        if bootstrap:
            metrics = MetricCollection({
                "f1": BootStrapper(F1Score(num_classes=num_classes, **self.f1_params), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
                "auroc": BootStrapper(AUROC(num_classes=num_classes, **self.auroc_params), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
                "precision": BootStrapper(Precision(num_classes=num_classes, **self.precision_params), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
                "recall": BootStrapper(Recall(num_classes=num_classes, **self.recall_params), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
            })
        else:
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
            self.test_metrics = self._init_metrics(pl_module.num_classes, pl_module.device, bootstrap=True)
            
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
                if name.endswith("_std"):
                    var_name = name[:-4] + "_var"
                    var_value = value ** 2
                    pl_module.log(f"test/{var_name}", var_value, on_step=False, on_epoch=True, prog_bar=True)
            self.test_metrics.reset()


class RegressionMetricsCallback(Callback):
    """A Lightning callback that tracks additional regression metrics (MSE, MAE).
    
    This callback updates metrics at each batch and logs them via the LightningModule 
    at the end of validation and test epochs.

    Args:
        frequency (int): How often to print/log the non-minimum metrics table (e.g., 5 means every 5 epochs).
    """

    def __init__(self, frequency: int = 5, num_bootstraps: int = 200, sampling_strategy: str = "poisson") -> None:
        """Initializes the RegressionMetricsCallback.

        Args:
            frequency (int): How often to print/log the non-minimum metrics table (e.g., 5 means every 5 epochs).
                Defaults to 5 (every 5 epochs).
            num_bootstraps (int): The number of bootstrap resamples to use in the test stage.
                Defaults to 200.
            sampling_strategy (str): The sampling strategy to use for BootStrapper ('poisson' or 'multinomial').
                Defaults to "poisson".
        """
        super().__init__()
        self.frequency = frequency
        self.num_bootstraps = num_bootstraps
        self.sampling_strategy = sampling_strategy
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        
        # Track the best (lowest) validation MSE over epochs
        self.val_mse_best = MinMetric()

        # Lists to collect targets and predictions for non-minimum metric calculations
        self.val_preds: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.test_preds: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

    def _init_metrics(self, device: torch.device, bootstrap: bool = False) -> MetricCollection:
        """Initializes the MetricCollection with MSE and MAE.

        Args:
            device (torch.device): The device (CPU/GPU) where metrics should be allocated.
            bootstrap (bool): Whether to wrap metrics in a BootStrapper. Defaults to False.

        Returns:
            MetricCollection: A collection of initialized torchmetrics.
        """
        from torchmetrics import MeanSquaredError, MeanAbsoluteError
        if bootstrap:
            metrics = MetricCollection({
                "mse": BootStrapper(MeanSquaredError(), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
                "mae": BootStrapper(MeanAbsoluteError(), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy),
            })
        else:
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
        self.val_preds = []
        self.val_targets = []

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
            
            # Collect for epoch-end non-minimum metrics calculation only on matching epochs
            if trainer.current_epoch % self.frequency == 0:
                self.val_preds.append(preds.detach().cpu())
                self.val_targets.append(targets.detach().cpu())

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

        if trainer.current_epoch % self.frequency != 0:
            return

        if self.val_preds:
            all_preds = torch.cat(self.val_preds)
            all_targets = torch.cat(self.val_targets)
            
            min_target = all_targets.min().item()
            mask = all_targets > min_target
            
            if mask.any():
                non_min_preds = all_preds[mask]
                non_min_targets = all_targets[mask]
                
                mse_non_min = torch.mean((non_min_preds - non_min_targets) ** 2)
                mae_non_min = torch.mean(torch.abs(non_min_preds - non_min_targets))
                
                # Log to pl_module
                pl_module.log("val/mse_non_min", mse_non_min, on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("val/mae_non_min", mae_non_min, on_step=False, on_epoch=True, prog_bar=True)
                
                # Print to stdout
                count_non_min = mask.sum().item()
                total_count = len(all_targets)
                msg = (
                    f"\n=== Regression Metrics for Targets > {min_target:.4f} (Validation Epoch {trainer.current_epoch}) ===\n"
                    f"Samples: {count_non_min}/{total_count} ({count_non_min/total_count*100:.1f}%)\n"
                    f"MSE: {mse_non_min.item():.4f}\n"
                    f"MAE: {mae_non_min.item():.4f}\n"
                    f"=========================================================================\n"
                )
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except ImportError:
                    print(msg)
            else:
                msg = (
                    f"\n=== Regression Metrics (Validation Epoch {trainer.current_epoch}) ===\n"
                    f"All validation targets are equal to the minimum value: {min_target:.4f}\n"
                    f"Cannot calculate metrics for targets > min_target.\n"
                    f"=========================================================================\n"
                )
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except ImportError:
                    print(msg)
            
            # Save non-minimum metrics to a file in the log directory
            import os
            log_dir = None
            if hasattr(trainer, "log_dir") and isinstance(trainer.log_dir, str):
                log_dir = trainer.log_dir
            elif hasattr(trainer, "default_root_dir") and isinstance(trainer.default_root_dir, str):
                log_dir = trainer.default_root_dir

            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "regression_metrics.txt")
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            
            self.val_preds = []
            self.val_targets = []

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Resets metrics at the beginning of the test epoch."""
        if self.test_metrics is not None:
            self.test_metrics.reset()
        self.test_preds = []
        self.test_targets = []

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
            self.test_metrics = self._init_metrics(pl_module.device, bootstrap=True)
            
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            preds = outputs["preds"].squeeze()
            targets = outputs["targets"].squeeze()
            
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
                
            self.test_metrics.update(preds, targets)

            # Collect for epoch-end non-minimum metrics calculation
            self.test_preds.append(preds.detach().cpu())
            self.test_targets.append(targets.detach().cpu())

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes and logs test metrics at the end of the epoch."""
        overall_msg = ""
        if self.test_metrics is not None:
            output = self.test_metrics.compute()
            # Log all metrics in the collection
            for name, value in output.items():
                pl_module.log(f"test/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
                if name.endswith("_std"):
                    var_name = name[:-4] + "_var"
                    var_value = value ** 2
                    pl_module.log(f"test/{var_name}", var_value, on_step=False, on_epoch=True, prog_bar=True)
            
            # Extract standard metrics for stdout/file logging
            mse_mean = output.get("mse_mean")
            mse_std = output.get("mse_std")
            mae_mean = output.get("mae_mean")
            mae_std = output.get("mae_std")
            
            if mse_mean is not None and mse_std is not None and mae_mean is not None and mae_std is not None:
                overall_msg = (
                    f"\n=== Overall Regression Metrics (Test Epoch) ===\n"
                    f"MSE Mean: {mse_mean.item():.4f}, Std: {mse_std.item():.4f}, Var: {(mse_std**2).item():.4f}\n"
                    f"MAE Mean: {mae_mean.item():.4f}, Std: {mae_std.item():.4f}, Var: {(mae_std**2).item():.4f}\n"
                    f"================================================\n"
                )
                try:
                    from tqdm import tqdm
                    tqdm.write(overall_msg)
                except ImportError:
                    print(overall_msg)
                    
            self.test_metrics.reset()

        msg = ""
        if self.test_preds:
            all_preds = torch.cat(self.test_preds)
            all_targets = torch.cat(self.test_targets)
            
            min_target = all_targets.min().item()
            mask = all_targets > min_target
            
            if mask.any():
                non_min_preds = all_preds[mask].to(pl_module.device)
                non_min_targets = all_targets[mask].to(pl_module.device)
                
                from torchmetrics import MeanSquaredError, MeanAbsoluteError
                
                bs_mse = BootStrapper(MeanSquaredError(), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy).to(pl_module.device)
                bs_mae = BootStrapper(MeanAbsoluteError(), num_bootstraps=self.num_bootstraps, sampling_strategy=self.sampling_strategy).to(pl_module.device)
                
                bs_mse.update(non_min_preds, non_min_targets)
                bs_mae.update(non_min_preds, non_min_targets)
                
                res_mse = bs_mse.compute()
                res_mae = bs_mae.compute()
                
                mse_non_min_mean = res_mse["mean"]
                mse_non_min_std = res_mse["std"]
                mse_non_min_var = mse_non_min_std ** 2
                
                mae_non_min_mean = res_mae["mean"]
                mae_non_min_std = res_mae["std"]
                mae_non_min_var = mae_non_min_std ** 2
                
                # Log to pl_module
                pl_module.log("test/mse_non_min_mean", mse_non_min_mean, on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/mse_non_min_std", mse_non_min_std, on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/mse_non_min_var", mse_non_min_var, on_step=False, on_epoch=True, prog_bar=True)
                
                pl_module.log("test/mae_non_min_mean", mae_non_min_mean, on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/mae_non_min_std", mae_non_min_std, on_step=False, on_epoch=True, prog_bar=True)
                pl_module.log("test/mae_non_min_var", mae_non_min_var, on_step=False, on_epoch=True, prog_bar=True)
                
                # Print to stdout
                count_non_min = mask.sum().item()
                total_count = len(all_targets)
                msg = (
                    f"\n=== Regression Metrics for Targets > {min_target:.4f} (Test Epoch) ===\n"
                    f"Samples: {count_non_min}/{total_count} ({count_non_min/total_count*100:.1f}%)\n"
                    f"MSE Mean: {mse_non_min_mean.item():.4f}, Std: {mse_non_min_std.item():.4f}, Var: {mse_non_min_var.item():.4f}\n"
                    f"MAE Mean: {mae_non_min_mean.item():.4f}, Std: {mae_non_min_std.item():.4f}, Var: {mae_non_min_var.item():.4f}\n"
                    f"=========================================================================\n"
                )
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except ImportError:
                    print(msg)
            else:
                msg = (
                    f"\n=== Regression Metrics (Test Epoch) ===\n"
                    f"All test targets are equal to the minimum value: {min_target:.4f}\n"
                    f"Cannot calculate metrics for targets > min_target.\n"
                    f"=========================================================================\n"
                )
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except ImportError:
                    print(msg)
            
            # Save metrics to a file in the log directory
            import os
            log_dir = None
            if hasattr(trainer, "log_dir") and isinstance(trainer.log_dir, str):
                log_dir = trainer.log_dir
            elif hasattr(trainer, "default_root_dir") and isinstance(trainer.default_root_dir, str):
                log_dir = trainer.default_root_dir

            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "regression_metrics.txt")
                with open(log_file_path, "a", encoding="utf-8") as f:
                    if overall_msg:
                        f.write(overall_msg)
                    f.write(msg)

            self.test_preds = []
            self.test_targets = []


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



