"""Catch22 feature engineering for tabular tree models.

Converts each modality's time-series window into the 22 catch22 features
using the pycatch22 package.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

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
        exclude_last_n_cols: Union[int, str, None] = "auto",
        channel_mode: str = "global",
        num_modalities: Optional[int] = None,
    ) -> None:
        if channel_mode not in ("all", "global", "subject"):
            raise ValueError(f"Unknown channel_mode '{channel_mode}'. Supported: 'all', 'global', 'subject'")
        self.fill_value = fill_value
        self.exclude_last_n_cols = exclude_last_n_cols
        self.channel_mode = channel_mode
        self.num_modalities = num_modalities

        # Get the catch22 feature names from pycatch22.
        # pycatch22 C library requires >= 3 points; 2 points causes a segfault.
        dummy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.stats = pycatch22.catch22_all(dummy)["names"]

    def _select_channels(self, x: np.ndarray) -> np.ndarray:
        """Selects modality channels according to channel_mode and num_modalities.

        DualScaler outputs channels in contiguous modality blocks:
        [global_mod_0..global_mod_{M-1}, subject_mod_0..subject_mod_{M-1}].
        When channel_mode is 'global' or 'subject', this extracts the corresponding
        M-column block, avoiding duplicate Catch22 calculations while preserving all
        distinct sensor modalities in their original order.
        """
        if self.channel_mode == "all":
            return x

        n_cols = x.shape[-1]
        m = self.num_modalities

        if m is not None:
            if n_cols == 2 * m:
                # DualScaler is present: first m cols are global, next m are subject
                if self.channel_mode == "global":
                    return x[:, :, :m]
                elif self.channel_mode == "subject":
                    return x[:, :, m:2 * m]
            elif n_cols == m:
                # Single scaler or unscaled: already 1 stream per modality, keep all
                return x
            else:
                # If n_cols doesn't match 2*m or m, fallback safely if evenly doubled
                if n_cols % 2 == 0 and n_cols > 0:
                    mid = n_cols // 2
                    return x[:, :, :mid] if self.channel_mode == "global" else x[:, :, mid:2 * mid]
                return x
        else:
            # If num_modalities was not specified, infer from even split if doubled
            if n_cols % 2 == 0 and n_cols > 0:
                mid = n_cols // 2
                return x[:, :, :mid] if self.channel_mode == "global" else x[:, :, mid:2 * mid]
            return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # Accept either one sample [Time, Features] or a batch [N, Time, Features].
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected a [Time, Features] or [N, Time, Features] array, got shape {x.shape}")

        # Drop the last exclude_last_n_cols columns (0 = keep everything).
        # If 'auto' or None was not resolved by FLAMLHealthModule, default to 0.
        exclude_cols = 0 if self.exclude_last_n_cols in (None, "auto") else int(self.exclude_last_n_cols)
        n_cols = x.shape[-1] - exclude_cols
        if n_cols <= 0:
            raise ValueError(
                f"exclude_last_n_cols={self.exclude_last_n_cols} leaves no feature columns "
                f"in an input of shape {x.shape}"
            )
        x = x[:, :, :n_cols].astype(np.float64)

        # Select channels based on channel_mode (e.g. deduplicate DualScaler)
        x = self._select_channels(x)

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

        out = out.reshape(n, f * n_feats)
        return out[0] if squeeze else out

    def feature_names(self, column_names: Optional[Sequence[str]] = None, n_features: Optional[int] = None) -> List[str]:
        """Names transform()'s output columns; never affects what transform() computes."""
        names = list(column_names or [])
        total = n_features if n_features is not None else len(names)
        if len(names) < total:
            names = names + [f"col{i}" for i in range(len(names), total)]
        names = names[:total]

        # Filter column names if channel_mode selected a subset
        if self.channel_mode in ("global", "subject") and names:
            m = self.num_modalities
            if m is not None:
                if len(names) == 2 * m:
                    names = names[:m] if self.channel_mode == "global" else names[m:2 * m]
            elif len(names) % 2 == 0 and len(names) > 0:
                mid = len(names) // 2
                names = names[:mid] if self.channel_mode == "global" else names[mid:2 * mid]

        return [f"{name}__{stat}" for name in names for stat in self.stats]
