"""Summary-statistics feature engineering for tabular tree models.

Converts a time-series sensor window into a fixed set of
summary statistics for each modality.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

#These functions calculate one statistic across the time steps.
# The input has shape [N, T]:
# N = number of samples
# T = number of time steps
_STATS = {
    "mean": lambda c: np.mean(c, axis=1),
    "median": lambda c: np.median(c, axis=1),
    "std": lambda c: np.std(c, axis=1),
    "min": lambda c: np.min(c, axis=1),
    "max": lambda c: np.max(c, axis=1),
}

#stats to compute by default, and the order they are in the output.
DEFAULT_STATS: List[str] = ["mean", "median", "std", "min", "max"]

# name of statistics
SUPPORTED_STATS = tuple(_STATS)


class SummaryStatsFeaturizer:
    """Collapses ``[N, Time, Features]`` sensor windows into summary stats."""

    def __init__(
        self,
        stats: Sequence[str] = DEFAULT_STATS,
        exclude_last_n_cols: Union[int, str, None] = "auto",
    ) -> None:
        # Check that the requested statistics are supported.
        unknown = sorted(set(stats) - set(SUPPORTED_STATS))
        ## If a statistic is not supported, stop and show an error.
        if unknown:
            raise ValueError(f"Unknown stat(s) {unknown}; supported: {sorted(SUPPORTED_STATS)}")
        self.stats = list(stats)   #save the statistics
        # Trailing columns to drop before computing stats (e.g. a sampler's
        # appended sin/cos time-of-day columns, which aren't modality signal).
        # Can be an integer, 'auto', or None.
        self.exclude_last_n_cols = exclude_last_n_cols

    def transform(self, x: np.ndarray) -> np.ndarray:
        #input as a numpy array
        x = np.asarray(x)
        # Check if we received one sample instead of a batch.
        # One sample: [Time, Features]
        # Batch:      [N, Time, Features]
        squeeze = x.ndim == 2
        #If we have one sample, add a batch dimension.
        if squeeze:
            x = x[None, ...]
            # Make sure the input is either 2D or 3D
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
        # Keep only the feature/modality columns we want.
        # Convert them to float64 for the calculations.
        x = x[:, :, :n_cols].astype(np.float64)


        # Get the dimensions of the data.
        # n = number of samples
        # t = number of time steps
        # f = number of features/modalities
        n, t, f = x.shape  # samples, time steps, modality/feature columns
        # # Create an empty array to store the calculated statistics.
        # Shape: [samples, features, statistics]
        out = np.zeros((n, f, len(self.stats)), dtype=np.float32)

        # An empty window (0 time steps, e.g. an empty data split) leaves
        # every stat at 0.0 instead of computing mean([]) etc.
        # Only calculate statistics if there are time steps.
        if t > 0:
            ## Go through each feature/modality.
            for col in range(f):
                # Get the values for this feature across all time steps.
                # Shape: [N, T]
                channel = x[:, :, col]  # this channel's values, all samples: [N, T]
                ## Go through each statistic we want to calculate.
                for s_idx, stat_name in enumerate(self.stats):
                    # # Calculate the statistic for every sample.
                    # The calculation is across the time dimension.
                    out[:, col, s_idx] = _STATS[stat_name](channel)

        # Change the shape from:
        # [N, Features, Statistics]
        # to:
        # [N, Features × Statistics]
        # [N, F, S] -> [N, F*S], columns ordered [chan0_stat0, chan0_stat1, ..., chan1_stat0, ...]
        out = out.reshape(n, f * len(self.stats))
        # If the original input was one sample,
        # remove the batch dimension we added earlier.
        return out[0] if squeeze else out  # undo the batch dim if input was unbatched

    def feature_names(
        self,
        column_names: Optional[Sequence[str]] = None,
        n_features: Optional[int] = None,
    ) -> List[str]:
        """Return names for the features produced by transform()."""

        # Start with the provided feature names.
        # If none were provided, use an empty list.
        names = list(column_names or [])

        # Decide how many feature names we need.
        total = n_features if n_features is not None else len(names)

        # If we don't have enough names, create generic names.
        if len(names) < total:
            names = names + [
                f"col{i}" for i in range(len(names), total)
            ]

        # Keep only the number of names we need.
        names = names[:total]

        # Create a name for every feature/statistic combination.
        #
        # Example:
        # step__mean
        # step__median
        # step__std
        # step__min
        # step__max
        return [
            f"{name}__{stat}"
            for name in names
            for stat in self.stats
        ]