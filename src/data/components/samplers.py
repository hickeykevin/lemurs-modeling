import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Dict, List, Tuple

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

class RollingSampler(TimeSampler):
    """New logic: Samples X hours exactly preceding the survey timestamp."""
    
    def __init__(self, lookback_hours: float = 24.0, resample_freq: str = "1h"):
        self.lookback_hours = lookback_hours
        self.resample_freq = resample_freq
        
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
            
        return np.stack(all_modality_features, axis=-1)

class OffsetSampler(TimeSampler):
    """Samples data based on fixed offsets from the midnight of the survey day.
    
    Offsets are in hours relative to 00:00 (midnight) of the survey date.
    
    Examples:
        - Previous Calendar Day (00:00 to 23:59 yesterday):
            start_offset_hours = -24
            end_offset_hours = 0
            
        - Today's Work Day (9:00 AM to 5:00 PM today):
            start_offset_hours = 9
            end_offset_hours = 17
            
        - Overnight / Sleep Window (6:00 PM yesterday to 6:00 AM today):
            start_offset_hours = -6
            end_offset_hours = 6
            
        - Morning of Survey (00:00 AM to 9:00 AM today):
            start_offset_hours = 0
            end_offset_hours = 9
    """
    
    def __init__(self, start_offset_hours: float = -24.0, end_offset_hours: float = 0.0, resample_freq: str = "1h"):
        self.start_offset_hours = start_offset_hours
        self.end_offset_hours = end_offset_hours
        self.resample_freq = resample_freq
        
    def __call__(self, survey_timestamp, app_user_id, modality_dfs, modality_cols, modalities):
        # Anchor to midnight of the survey day
        day_start = pd.Timestamp(survey_timestamp.date())
        
        start_time = day_start + timedelta(hours=self.start_offset_hours)
        end_time = day_start + timedelta(hours=self.end_offset_hours)
        
        duration_hours = (end_time - start_time).total_seconds() / 3600
        num_periods = int(pd.Timedelta(hours=duration_hours) / pd.Timedelta(self.resample_freq))
        
        full_range = pd.date_range(
            start=start_time, 
            periods=num_periods, 
            freq=self.resample_freq
        )
        
        all_modality_features = []
        for mod in modalities:
            df = modality_dfs[mod]
            val_col = modality_cols[mod]
            
            mask = (df['app_user_id'] == app_user_id) & \
                   (df['start_timestamp'] >= start_time) & \
                   (df['start_timestamp'] < end_time)
            
            window_data = df[mask].copy()

            if window_data.empty:
                resampled = np.zeros(len(full_range), dtype=np.float32)
            else:
                window_data['bin'] = window_data['start_timestamp'].dt.floor(self.resample_freq)
                resampled = (window_data.groupby('bin')[val_col]
                             .sum()
                             .reindex(full_range, fill_value=0)
                             .values.astype(np.float32))
            all_modality_features.append(resampled)
            
        return np.stack(all_modality_features, axis=-1)
