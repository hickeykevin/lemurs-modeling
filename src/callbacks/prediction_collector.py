from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer


class PredictionCollectorCallback(Callback):
    """Collects per-sample test predictions, joined back to their source row.

    ``HealthLitModule.test_step`` (like ``validation_step``) returns a dict of
    batch-level tensors (``preds``, ``targets``, ``logits``) with no way to
    trace a prediction back to which participant or timestamp it came from --
    exactly what pooling predictions across a per-user walk-forward CV's folds
    requires (see ``WalkForwardHealthDataModule``): a prediction is only
    meaningful once it is known whose forecast, and of what time, it is.

    This reads the sample index from the raw batch (the last tensor, when the
    active dataset was built with ``return_index=True`` -- see
    ``HealthDataset``) rather than from ``outputs``, since
    ``HealthLitModule.model_step`` has no notion of an index: it destructures
    the batch by fixed length (4/5/6) and would silently misread an appended
    idx tensor as the demographics tensor. Reading ``batch`` directly,
    independent of whatever ``model_step`` does with the same batch,
    sidesteps that rather than requiring any change to ``HealthLitModule``.

    Whether the batch's last element is actually an index (as opposed to a
    demographics tensor a length-4/5/6 batch might otherwise end with) cannot
    be inferred from shape alone, so this checks
    ``trainer.datamodule.data_test``'s (or ``data_val``'s) own
    ``return_index`` flag rather than guessing -- and raises if it is not set,
    since a silent no-op collector is worse than a loud one.

    Usage: attach a fresh instance per fold (state is not reset between
    stages/folds automatically), call ``trainer.test(...)``, then read
    ``.rows`` -- a list of dicts with ``fold_index``, ``app_user_id``,
    ``record_timestamp``, ``y_true``, and one ``prob_class_{k}`` column per
    class. An evaluation plan (e.g. ``WalkForwardPlan``) concatenates
    ``.rows`` across folds into one pooled prediction table.
    """

    def __init__(self, fold_index: Optional[int] = None) -> None:
        """Initializes the PredictionCollectorCallback.

        Args:
            fold_index: Tag attached to every row this instance collects, so
                a caller pooling across several instances (one per fold) can
                tell which fold a prediction came from without tracking it
                separately. The column is always present in ``.rows``/
                ``.to_dataframe()`` for a consistent schema; leave this
                ``None`` for a single-stage use with no fold concept.
        """
        super().__init__()
        self.fold_index = fold_index
        self.rows: List[Dict[str, Any]] = []

    def _dataset_for(self, trainer: Trainer, stage: str):
        dm = getattr(trainer, "datamodule", None)
        return getattr(dm, f"data_{stage}", None) if dm is not None else None

    def _collect(
        self,
        trainer: Trainer,
        outputs: Optional[Dict[str, torch.Tensor]],
        batch: Any,
        stage: str,
    ) -> None:
        if outputs is None or "targets" not in outputs or "logits" not in outputs:
            return

        dataset = self._dataset_for(trainer, stage)
        if dataset is None or not getattr(dataset, "return_index", False):
            raise RuntimeError(
                "PredictionCollectorCallback requires the active "
                f"data_{stage} dataset to be built with return_index=True "
                "(see HealthDataset / WalkForwardHealthDataModule) so "
                "predictions can be joined back to app_user_id/"
                "record_timestamp. Without it, this callback would either "
                "silently collect nothing useful or misread an unrelated "
                "batch element as the index."
            )

        idx_tensor = batch[-1]
        if not torch.is_tensor(idx_tensor) or idx_tensor.dtype != torch.long:
            raise RuntimeError(
                f"Expected the last batch element to be a long index tensor "
                f"(return_index=True's contract), got {type(idx_tensor)} / "
                f"{getattr(idx_tensor, 'dtype', None)}. The batch shape may "
                "not match what return_index=True produces any more -- check "
                "HealthDataset.__getitem__."
            )

        idx = idx_tensor.detach().cpu().numpy()
        targets = outputs["targets"].detach().cpu().numpy()
        probs = torch.softmax(outputs["logits"].detach().cpu().float(), dim=-1).numpy()
        links = dataset.data_links

        for i, sample_idx in enumerate(idx):
            row = links.iloc[int(sample_idx)]
            entry = {
                "fold_index": self.fold_index,
                "stage": stage,
                "sample_idx": int(sample_idx),
                "app_user_id": row["app_user_id"],
                "record_timestamp": row["record_timestamp"],
                "y_true": int(targets[i]),
            }
            for k in range(probs.shape[-1]):
                entry[f"prob_class_{k}"] = float(probs[i, k])
            self.rows.append(entry)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[Dict[str, torch.Tensor]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(trainer, outputs, batch, "test")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[Dict[str, torch.Tensor]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        self._collect(trainer, outputs, batch, "val")

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the collected rows as a DataFrame, empty-safe."""
        if not self.rows:
            return pd.DataFrame(
                columns=["fold_index", "stage", "sample_idx", "app_user_id", "record_timestamp", "y_true"]
            )
        return pd.DataFrame(self.rows)
