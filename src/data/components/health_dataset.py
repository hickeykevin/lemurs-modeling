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
        demographics_map: Optional[Dict[Any, np.ndarray]] = None,
        default_demographics: Optional[np.ndarray] = None,
        use_sleep: bool = False,
        use_survey_context: bool = False,
    ) -> None:
        """Initializes the HealthDataset."""
        self.data_links = linked_data.reset_index(drop=True)
        self.sampler = sampler
        self.scaler = scaler
        self.modalities = sorted(list(modality_dfs.keys()))
        self.user_to_idx = user_to_idx
        self.is_regression = is_regression
        self.demographics_map = demographics_map
        self.default_demographics = default_demographics
        self.use_sleep = use_sleep
        self.use_survey_context = use_survey_context

        if self.use_survey_context:
            context_cols = ["is_morning", "referent_hours_scaled", "referent_missing"]
            for col in context_cols:
                if col not in self.data_links.columns:
                    self.data_links[col] = 1.0 if col == "referent_missing" else 0.0
            self.context_features = self.data_links[context_cols].values.astype(np.float32)

        if self.use_sleep:
            sleep_cols = ["sleep_hours_scaled", "sleep_class_0", "sleep_class_1", "sleep_class_2", "sleep_unknown"]
            for col in sleep_cols:
                if col not in self.data_links.columns:
                    if col == "sleep_unknown":
                        self.data_links[col] = 1.0
                    else:
                        self.data_links[col] = 0.0
            self.sleep_features = self.data_links[sleep_cols].values.astype(np.float32)

        self._sequences, self._targets, self._user_indices = self._precompute(modality_dfs, modality_cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute(
        self, modality_dfs: Dict[str, pd.DataFrame], modality_cols: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences: List[np.ndarray] = []
        targets: List[float] = []
        user_indices: List[int] = []

        timestamps = list(self.data_links["record_timestamp"])
        user_ids = self.data_links["app_user_id"].values
        answers = self.data_links["answer"].values

        for ts, uid, ans in zip(timestamps, user_ids, answers):
            seq = self.sampler(
                survey_timestamp=ts,
                app_user_id=uid,
                modality_dfs=modality_dfs,
                modality_cols=modality_cols,
                modalities=self.modalities,
            )
            sequences.append(seq)
            if self.is_regression:
                targets.append(float(ans))
            else:
                targets.append(int(ans))

            # Map user ID to integer index
            if self.user_to_idx is not None and uid in self.user_to_idx:
                user_indices.append(self.user_to_idx[uid])
            else:
                user_indices.append(0)

        if not sequences:
            # An empty split (e.g. every response for every user fell within
            # a longitudinal purge gap) has no sequence to infer [T, F] from.
            # Returning a [0, T, F]-shaped-as-[0] array rather than crashing
            # lets the dataset behave as a normal, simply empty, Dataset.
            target_dtype = np.float32 if self.is_regression else np.int64
            return (
                np.zeros((0,), dtype=np.float32),
                np.array([], dtype=target_dtype),
                np.array([], dtype=np.int64),
            )

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
        ret = (
            torch.from_numpy(self._sequences[idx]),
            torch.tensor(self._targets[idx], dtype=target_dtype),
            torch.tensor(self._user_indices[idx], dtype=torch.long),
        )

        if self.demographics_map is not None or self.use_sleep or self.use_survey_context:
            parts = []
            if self.demographics_map is not None:
                uid = self.data_links.iloc[idx]["app_user_id"]
                parts.append(self.demographics_map.get(uid, self.default_demographics))
            if self.use_sleep:
                parts.append(self.sleep_features[idx])
            if self.use_survey_context:
                parts.append(self.context_features[idx])
            demo = np.concatenate(parts).astype(np.float32)
            ret = ret + (torch.tensor(demo, dtype=torch.float32),)

        return ret
