import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

class TimeSampler(ABC):
    """Base class for sampling health metrics relative to a survey timestamp."""
    
    @abstractmethod
    def __call__(
        self,
        survey_timestamp: pd.Timestamp,
        app_user_id: int,
        modality_dfs: Dict[str, pd.DataFrame],
        modality_cols: Dict[str, str],
        modalities: List[str]
    ) -> np.ndarray:
        """Slices and resamples data into a feature matrix.

        Returns:
            np.ndarray: Matrix of shape [Time, Features]
        """
        pass

    def window_bounds(
        self, survey_timestamp: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Returns the ``[start, end)`` span this sampler reads for a survey.

        Used to determine whether a sample has any sensor coverage at all,
        without running the full sampling path. Samplers that do not read
        sensor data (e.g. ``LagSampler``) return ``None``.
        """
        return None

    @property
    def num_time_features(self) -> int:
        """Number of cyclic time features appended to the end of the sampled array."""
        return 0

class RollingSampler(TimeSampler):
    """New logic: Samples X hours exactly preceding the survey timestamp."""
    
    def __init__(self, lookback_hours: float = 24.0, resample_freq: str = "1h", include_time_features: bool = True, **kwargs):
        self.lookback_hours = lookback_hours
        self.resample_freq = resample_freq
        self.include_time_features = include_time_features

    @property
    def num_time_features(self) -> int:
        """Appends sin/cos hour and sin/cos day-of-week (4 columns)."""
        return 4 if self.include_time_features else 0

    def window_bounds(self, survey_timestamp):
        end_time = pd.Timestamp(survey_timestamp).floor(self.resample_freq)
        return end_time - timedelta(hours=self.lookback_hours), end_time

    def __call__(self, survey_timestamp, app_user_id, modality_dfs, modality_cols, modalities):
        end_time = survey_timestamp.floor(self.resample_freq)
        start_time = end_time - timedelta(hours=self.lookback_hours)
        
        # Determine number of periods
        num_periods = int(pd.Timedelta(hours=self.lookback_hours) / pd.Timedelta(self.resample_freq))
        
        # We want the bins to start at start_time and cover num_periods.
        # For a 12h lookback ending at 12:00, start is 00:00. 
        # Bins: 00:00, 01:00, ..., 11:00 (12 bins total).
        full_range = pd.date_range(
            start=start_time, 
            periods=num_periods, 
            freq=self.resample_freq
        )
        
        all_modality_features = []
        for mod in modalities:
            df = modality_dfs[mod]
            val_col = modality_cols[mod]
            
            # Filter for user and the specific time range
            mask = (df['app_user_id'] == app_user_id) & \
                   (df['start_timestamp'] >= start_time) & \
                   (df['start_timestamp'] < end_time)
            
            window_data = df[mask].copy()

            if window_data.empty:
                resampled = np.zeros(len(full_range), dtype=np.float32)
            else:
                # Resample based on the provided frequency
                window_data['bin'] = window_data['start_timestamp'].dt.floor(self.resample_freq)
                resampled = (window_data.groupby('bin')[val_col]
                             .sum()
                             .reindex(full_range, fill_value=0)
                             .values.astype(np.float32))
            all_modality_features.append(resampled)
            
        # Append cyclic time features
        if self.include_time_features:
            hours = full_range.hour.values
            sin_hour = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
            cos_hour = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
            
            weekdays = full_range.weekday.values
            sin_weekday = np.sin(2 * np.pi * weekdays / 7.0).astype(np.float32)
            cos_weekday = np.cos(2 * np.pi * weekdays / 7.0).astype(np.float32)
            
            all_modality_features.append(sin_hour)
            all_modality_features.append(cos_hour)
            all_modality_features.append(sin_weekday)
            all_modality_features.append(cos_weekday)
            
        return np.stack(all_modality_features, axis=-1)
