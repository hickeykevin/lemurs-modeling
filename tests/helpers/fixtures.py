"""Test-only fixtures: a midnight-anchored sampler and simple mean/max label aggregators.

They give small hand-built cohorts deterministic windows and labels. The paper's runs use
``RollingSampler`` and ``RuleBasedAggregator`` (see ``src/data/components``).
"""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler


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
    
    def __init__(self, start_offset_hours: float = -24.0, end_offset_hours: float = 0.0, resample_freq: str = "1h", include_time_features: bool = True, **kwargs):
        self.start_offset_hours = start_offset_hours
        self.end_offset_hours = end_offset_hours
        self.resample_freq = resample_freq
        self.include_time_features = include_time_features

    @property
    def num_time_features(self) -> int:
        """Appends sin/cos hour (2 columns)."""
        return 2 if self.include_time_features else 0

    def window_bounds(self, survey_timestamp):
        day_start = pd.Timestamp(pd.Timestamp(survey_timestamp).date())
        return (
            day_start + timedelta(hours=self.start_offset_hours),
            day_start + timedelta(hours=self.end_offset_hours),
        )

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
            
        # Append cyclic time features.
        # NOTE: hour-of-day channels are intentionally omitted here. OffsetSampler
        # anchors every window to midnight + a fixed offset and steps by a fixed
        # resample_freq, so the bin hours are identical for every sample. That
        # makes sin/cos hour constant across samples (a fixed positional encoding)
        # with no cross-sample discriminative signal. Weekday still varies with
        # the calendar dates the window spans, so it is retained.
        if self.include_time_features:
            weekdays = full_range.weekday.values
            sin_weekday = np.sin(2 * np.pi * weekdays / 7.0).astype(np.float32)
            cos_weekday = np.cos(2 * np.pi * weekdays / 7.0).astype(np.float32)

            all_modality_features.append(sin_weekday)
            all_modality_features.append(cos_weekday)

        return np.stack(all_modality_features, axis=-1)


class MeanAggregator(LabelAggregator):
    """Aggregates answers by taking the mean and optionally binarizing."""
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None, **kwargs):
        self.question_ids = question_ids if question_ids is not None else [2, 3]
        self.threshold = threshold

    def get_question_ids(self) -> list:
        return self.question_ids
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].mean().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg

class MaxAggregator(LabelAggregator):
    """Aggregates answers by taking the max and optionally binarizing."""
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None, **kwargs):
        self.question_ids = question_ids if question_ids is not None else [2, 3]
        self.threshold = threshold

    def get_question_ids(self) -> list:
        return self.question_ids
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].max().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg
