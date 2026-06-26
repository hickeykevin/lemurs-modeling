import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from src.data.components.samplers import TimeSampler


class HealthDataset(Dataset):
    """General Multimodal Dataset for health metrics.

    All sequences are pre-computed at construction time so that ``__getitem__``
    is a simple indexed array lookup with zero Pandas overhead per epoch.

    If a scaler is provided, it is applied once to the full stacked array in a
    single vectorised sklearn call rather than one reshape-transform per sample.

    Attributes:
        data_links (pd.DataFrame): Survey-to-user linkage metadata.
        modality_dfs (Dict[str, pd.DataFrame]): Raw modality dataframes keyed by name.
        modality_cols (Dict[str, str]): Mapping of modality name to its value column.
        sampler (TimeSampler): Strategy for generating time-series sequences.
        scaler (Optional[Any]): Pre-fitted sklearn-compatible scaler.
        modalities (List[str]): Sorted modality names (ensures consistent feature order).
    """

    def __init__(
        self,
        linked_data: pd.DataFrame,
        modality_dfs: Dict[str, pd.DataFrame],
        modality_cols: Dict[str, str],
        sampler: TimeSampler,
        scaler: Optional[Any] = None,
        user_to_idx: Optional[Dict[str, int]] = None,
        is_regression: bool = False,
        use_prev_prediction: bool = False,
    ) -> None:
        """Initializes the HealthDataset.

        Args:
            linked_data (pd.DataFrame): Survey records with ``app_user_id`` and
                ``record_timestamp`` columns.
            modality_dfs (Dict[str, pd.DataFrame]): Health dataframes, e.g.
                ``{'step': df}``.
            modality_cols (Dict[str, str]): Mapping of modality name to its value column.
            sampler (TimeSampler): Sampling strategy (BlockSampler, Rolling, etc.).
            scaler (Optional[Any]): A pre-fitted scaler (e.g. ``StandardScaler``).
            user_to_idx (Optional[Dict[str, int]]): Mapping from app_user_id to user embedding index.
            is_regression (bool): Whether this dataset yields targets for a regression task.
        """
        self.data_links = linked_data.reset_index(drop=True)
        self.sampler = sampler
        self.scaler = scaler
        self.modalities = sorted(list(modality_dfs.keys()))
        self.user_to_idx = user_to_idx
        self.is_regression = is_regression
        self.use_prev_prediction = use_prev_prediction
        
        if self.use_prev_prediction:
            self.predictions_cache = np.zeros(len(self.data_links), dtype=np.float32)

        # Pre-compute all sequences and targets once at construction time.
        # __getitem__ then becomes a direct numpy array index — no Pandas overhead.
        self._sequences, self._targets, self._user_indices = self._precompute(modality_dfs, modality_cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute(self, modality_dfs: Dict[str, pd.DataFrame], modality_cols: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-computes all sequences and targets, applying the scaler once.

        Runs the sampler for every record in ``data_links`` and stacks the
        results. If a scaler is present it is applied to the full ``[N*T, F]``
        matrix in one call, which is far cheaper than N individual calls.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - sequences: ``float32`` array of shape ``[N, Time, Modalities]``.
                - targets: ``int64`` array of shape ``[N]``.
                - user_indices: ``int64`` array of shape ``[N]``.
        """
        sequences: List[np.ndarray] = []
        targets: List[float] = []
        user_indices: List[int] = []

        for idx in range(len(self.data_links)):
            row = self.data_links.iloc[idx]
            seq = self.sampler(
                survey_timestamp=row["record_timestamp"],
                app_user_id=row["app_user_id"],
                modality_dfs=modality_dfs,
                modality_cols=modality_cols,
                modalities=self.modalities,
            )
            sequences.append(seq)
            if self.is_regression:
                targets.append(float(row["answer"]))
            else:
                targets.append(int(row["answer"]))

            # Map user ID to integer index
            uid = row["app_user_id"]
            if self.user_to_idx is not None and uid in self.user_to_idx:
                user_indices.append(self.user_to_idx[uid])
            else:
                user_indices.append(0)

        seqs_np = np.stack(sequences, axis=0).astype(np.float32)  # [N, T, F]

        if self.scaler is not None:
            if hasattr(self.scaler, "transform_by_subject"):
                user_ids = self.data_links["app_user_id"].values
                seqs_np = self.scaler.transform_by_subject(seqs_np, user_ids)
            elif hasattr(self.scaler, "transform"):
                n, t, f = seqs_np.shape
                seqs_np = (
                    self.scaler.transform(seqs_np.reshape(-1, f))
                    .reshape(n, t, f)
                    .astype(np.float32)
                )

        target_dtype = np.float32 if self.is_regression else np.int64
        return seqs_np, np.array(targets, dtype=target_dtype), np.array(user_indices, dtype=np.int64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Returns the pre-computed sequence, target, and user index for index *idx*.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, ...]:
                - features: ``float32`` tensor of shape ``[Time, Modalities]``.
                - target: ``long`` or ``float32`` scalar tensor.
                - user_idx: ``long`` scalar tensor.
                - (optional) prev_pred: ``float32`` scalar tensor.
                - (optional) idx: ``long`` scalar tensor.
        """
        target_dtype = torch.float32 if self.is_regression else torch.long
        base_tuple = (
            torch.from_numpy(self._sequences[idx]),
            torch.tensor(self._targets[idx], dtype=target_dtype),
            torch.tensor(self._user_indices[idx], dtype=torch.long),
        )
        if self.use_prev_prediction:
            prev_idx = self.data_links.iloc[idx]["prev_sample_idx"]
            prev_pred = self.predictions_cache[prev_idx] if prev_idx != -1 else 0.0
            return base_tuple + (
                torch.tensor(prev_pred, dtype=torch.float32),
                torch.tensor(idx, dtype=torch.long),
            )
        return base_tuple
