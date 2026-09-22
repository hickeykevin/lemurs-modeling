"""Catch22 feature engineering for tabular tree models.

Converts each modality's time-series window into the 22 catch22 features
using the pycatch22 package.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pycatch22


class Catch22Featurizer:
    """Convert [N, Time, Features] sensor windows into catch22 features.

    Catch22 is applied separately to each modality across the full time
    window. The output contains 22 features per modality.
    """

    def __init__(
        self,
        fill_value: float = 0.0,
        exclude_last_n_cols: int = 0,
    ) -> None:
        self.fill_value = fill_value
        self.exclude_last_n_cols = exclude_last_n_cols

        # Get the catch22 feature names from pycatch22.
        # pycatch22 C library requires >= 3 points; 2 points causes a segfault.
        dummy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.stats = pycatch22.catch22_all(dummy)["names"]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # Accept either one sample [Time, Features] or a batch [N, Time, Features].
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected a [Time, Features] or [N, Time, Features] array, got shape {x.shape}")

        # Drop the last exclude_last_n_cols columns (0 = keep everything).
        n_cols = x.shape[-1] - self.exclude_last_n_cols
        if n_cols <= 0:
            raise ValueError(
                f"exclude_last_n_cols={self.exclude_last_n_cols} leaves no feature columns "
                f"in an input of shape {x.shape}"
            )
        x = x[:, :, :n_cols].astype(np.float64)

        n, t, f = x.shape  # samples, time steps, modality/feature columns
        n_feats = len(self.stats)
        # Store 22 catch22 features for each modality.
        out = np.zeros((n, f, n_feats), dtype=np.float32)

        if t >= 3:
            for row in range(n):
                for col in range(f):
                    series = x[row, :, col]  # this sample's, this modality's, full bin sequence
                    result = pycatch22.catch22_all(series)
                    values = np.asarray(result["values"], dtype=np.float64)
                    values = np.where(np.isfinite(values), values, self.fill_value)
                    out[row, col, :] = values
        elif t > 0:
            # Time series with t < 3 is too short for catch22 C calculations (requires >= 3 points)
            out.fill(self.fill_value)
        # else: t == 0 (an empty window, e.g. an empty data split) -- out
        # stays all zeros from np.zeros() above, same convention as
        # SummaryStatsFeaturizer.

        # For an empty time window, keep the initialized zero values.

        out = out.reshape(n, f * n_feats)
        return out[0] if squeeze else out

    def feature_names(self, column_names: Optional[Sequence[str]] = None, n_features: Optional[int] = None) -> List[str]:
        """Names transform()'s output columns; never affects what transform() computes."""
        names = list(column_names or [])
        total = n_features if n_features is not None else len(names)
        if len(names) < total:
            names = names + [f"col{i}" for i in range(len(names), total)]
        names = names[:total]
        return [f"{name}__{stat}" for name in names for stat in self.stats]
