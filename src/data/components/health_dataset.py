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
    ) -> None:
        """Initializes the HealthDataset.

        Args:
            linked_data (pd.DataFrame): Survey records with ``app_user_id`` and
                ``record_timestamp`` columns.
            modality_dfs (Dict[str, pd.DataFrame]): Health dataframes, e.g.
                ``{'step': df}``.
            modality_cols (Dict[str, str]): Value column names per modality, e.g.
                ``{'step': 'steps'}``.
            sampler (TimeSampler): Sampling strategy (BlockSampler, Rolling, etc.).
            scaler (Optional[Any]): A pre-fitted scaler (e.g. ``StandardScaler``).
        """
        self.data_links = linked_data.reset_index(drop=True)
        self.sampler = sampler
        self.scaler = scaler
        self.modalities = sorted(list(modality_dfs.keys()))

        # Pre-compute all sequences and targets once at construction time.
        # __getitem__ then becomes a direct numpy array index — no Pandas overhead.
        self._sequences, self._targets = self._precompute(modality_dfs, modality_cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute(self, modality_dfs: Dict[str, pd.DataFrame], modality_cols: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-computes all sequences and targets, applying the scaler once.

        Runs the sampler for every record in ``data_links`` and stacks the
        results. If a scaler is present it is applied to the full ``[N*T, F]``
        matrix in one call, which is far cheaper than N individual calls.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sequences: ``float32`` array of shape ``[N, Time, Modalities]``.
                - targets: ``int64`` array of shape ``[N]``.
        """
        sequences: List[np.ndarray] = []
        targets: List[int] = []

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
            targets.append(int(row["answer"]))

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

        return seqs_np, np.array(targets, dtype=np.int64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the pre-computed sequence and target for index *idx*.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - features: ``float32`` tensor of shape ``[Time, Modalities]``.
                - target: ``long`` scalar tensor.
        """
        return (
            torch.from_numpy(self._sequences[idx]),
            torch.tensor(self._targets[idx], dtype=torch.long),
        )
