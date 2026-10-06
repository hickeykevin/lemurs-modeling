"""Frozen TiRex embeddings as tabular features for the FLAML models (``model/featurizer=tirex``).

The tree-head counterpart of ``TiRexEncoderNet``: instead of a linear probe, the frozen
TiRex embedding of each modality's window is handed to the same FLAML estimators
(RF/XGBoost/LR/SVM) as the RQ2 ``summary_stats`` and ``catch22`` featurizers. With
``append_summary_stats=True`` the output is ``[TiRex embedding | SummaryStatsFeaturizer
output]``, so a comparison with the ``summary_stats`` runs isolates what the embedding adds.

Only the first ``num_modalities`` columns (the globally scaled values under ``DualScaler``)
are encoded: TiRex standardises every series itself, so the subject-scaled copy would give
the same embedding. The summary statistics use every non-time column, exactly as
``SummaryStatsFeaturizer`` does. Embeddings are cached per series (see
``TiRexEncoderNet.series_keys``), so FLAML's repeated transforms and CV folds that refit the
scaler reuse one encoding. Runs on GPU when available (about 16 min per 1,300 two-modality
windows on 8 CPU threads, a few seconds on an L40S).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from src.data.components.tabular_features import DEFAULT_STATS, SummaryStatsFeaturizer


class TiRexFeaturizer:
    """Convert ``[N, Time, Features]`` sensor windows into frozen TiRex embeddings (+ summary stats)."""

    def __init__(
        self,
        layer_pooling: str = "mean",
        append_summary_stats: bool = True,
        stats: Sequence[str] = DEFAULT_STATS,
        exclude_last_n_cols: Union[int, str, None] = "auto",
        num_modalities: Optional[int] = None,
        device: str = "auto",
        batch_size: int = 512,
        checkpoint_repo_id: str = "NX-AI/TiRex",
    ) -> None:
        self.layer_pooling = layer_pooling
        self.append_summary_stats = append_summary_stats
        self.stats = list(stats)
        self.exclude_last_n_cols = exclude_last_n_cols
        self.num_modalities = num_modalities
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_repo_id = checkpoint_repo_id
        self._net = None  # TiRexEncoderNet, built lazily for the encoder and its cache

    def _encoder(self, n_modalities: int):
        from src.models.components.tirex_net import TiRexEncoderNet

        if self._net is None or len(self._net.modalities) != n_modalities:
            net = TiRexEncoderNet(
                modalities=[f"m{i}" for i in range(n_modalities)],
                checkpoint_repo_id=self.checkpoint_repo_id,
                layer_pooling=self.layer_pooling,
                second_block="none",
                batch_size=self.batch_size,
            )
            device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            net._ensure_model(torch.device(device))
            self._net = net
        return self._net

    def _n_modalities(self, n_cols: int) -> int:
        m = self.num_modalities
        if m is None or m <= 0:
            return n_cols
        if n_cols not in (m, 2 * m):
            raise ValueError(f"{n_cols} sensor columns cannot be {m} modalities (x1, or x2 for a DualScaler)")
        return m

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected a [Time, Features] or [N, Time, Features] array, got shape {x.shape}")
        exclude = 0 if self.exclude_last_n_cols in (None, "auto") else int(self.exclude_last_n_cols)
        n_cols = x.shape[-1] - exclude
        if n_cols <= 0:
            raise ValueError(f"exclude_last_n_cols={self.exclude_last_n_cols} leaves no columns in shape {x.shape}")
        x = x[:, :, :n_cols].astype(np.float32)
        m = self._n_modalities(n_cols)

        parts = []
        if x.shape[0] == 0:
            per_mod = 512 * (12 if self.layer_pooling == "all" else 1)
            parts.append(np.zeros((0, per_mod * m), dtype=np.float32))
        else:
            net = self._encoder(m)
            series = torch.from_numpy(np.ascontiguousarray(x[:, :, :m].transpose(0, 2, 1)))  # [N, M, T]
            parts.append(net._embed(series).cpu().numpy().astype(np.float32))
        if self.append_summary_stats:
            parts.append(SummaryStatsFeaturizer(stats=self.stats, exclude_last_n_cols=0).transform(x).astype(np.float32))
        out = np.concatenate(parts, axis=1)
        return out[0] if squeeze else out

    def feature_names(self, column_names: Optional[Sequence[str]] = None, n_features: Optional[int] = None) -> List[str]:
        """Names for transform()'s output columns (embedding dims, then summary-stat names)."""
        names = list(column_names or [])
        total = n_features if n_features is not None else len(names)
        names = (names + [f"col{i}" for i in range(len(names), total)])[:total]
        m = self._n_modalities(len(names)) if names else (self.num_modalities or 0)
        per_mod = 512 * (12 if self.layer_pooling == "all" else 1)
        emb = [f"tirex_{i}" for i in range(per_mod * m)]
        stats = (
            SummaryStatsFeaturizer(stats=self.stats, exclude_last_n_cols=0).feature_names(names, len(names))
            if self.append_summary_stats
            else []
        )
        return emb + stats
