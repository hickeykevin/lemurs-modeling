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

class BlockSampler(TimeSampler):
    """Samples health metrics by grouping data into predefined behavioral time blocks.
    
    This sampler is designed to handle data collection where records may vary 
    from short bursts to multi-hour aggregations. Rather than forcing a 
    high-resolution grid (e.g., hourly), it maps data to 4 distinct behavioral 
    segments per day, preserving continuous sum totals without artificial variance.
    
    Defined Time Blocks:
        1. Sleep: 00:00 – 08:00 (8 Hours)
        2. Morning: 08:00 – 12:00 (4 Hours)
        3. Afternoon: 12:00 – 17:00 (5 Hours)
        4. Evening: 17:00 – 24:00 (7 Hours)
        
    Cross-Block Strategy:
        If a tracking interval crosses thresholds, steps are distributed by weighted 
        durations. Sleep blocks hold a probabilistic dampening weight of 0.05 (vs 1.0) 
        to reflect realistic behavior.
        
        Example: A 4-hr window spanning 10:00 PM to 02:00 AM holding 1,000 steps:
          - Overlap Evening (2h * 1.0) = 2.0 weighted hrs
          - Overlap Sleep (2h * 0.05) = 0.1 weighted hrs
          - Total = 2.1 weighted hrs. Evening gets 95.2%, Sleep gets 4.8%.
        
    Example:
        >>> sampler = BlockSampler(lookback_days=1)
        >>> features = sampler(survey_ts, user_id, dfs, cols, ['step'])
        >>> features.shape
        (4, 1)

        
    Args:
        lookback_days (int): Number of whole 24-hour days to sample preceding the survey.
    """

    
    def __init__(self, lookback_days: int = 1):
        self.lookback_days = lookback_days
        
    def __call__(self, survey_timestamp, app_user_id, modality_dfs, modality_cols, modalities):
        # Anchor processing to midnight of the day the survey was taken
        day_start = pd.Timestamp(survey_timestamp.date())
        
        # Hardcoded boundaries defining the 4 behavioral segments of a day
        block_offsets = [
            (timedelta(hours=0), timedelta(hours=8)),    # Sleep
            (timedelta(hours=8), timedelta(hours=12)),   # Morning
            (timedelta(hours=12), timedelta(hours=17)),  # Afternoon
            (timedelta(hours=17), timedelta(hours=24))   # Evening
        ]
        
        # Build consecutive window boundaries progressing chronologically
        blocks = []
        for d in range(self.lookback_days, 0, -1):
            base_date = day_start - timedelta(days=d)
            for b_start, b_end in block_offsets:
                blocks.append((base_date + b_start, base_date + b_end))
                
        # Outer boundaries bounding the full data extract requirement
        global_start = blocks[0][0]
        global_end = blocks[-1][1]
        
        all_modality_features = []
        for mod in modalities:
            df = modality_dfs[mod]
            val_col = modality_cols[mod]
            
            # Extract records applicable to the target user
            mask = (df['app_user_id'] == app_user_id) & \
                   (df['start_timestamp'] < global_end)
                   
            user_data = df[mask].copy()
            
            block_values = np.zeros(len(blocks), dtype=np.float32)
            
            if not user_data.empty:
                if 'end_timestamp' in user_data.columns:
                    # Apply a probabilistic weight to prevent over-allocating steps to sleep periods
                    weights = [0.05, 1.0, 1.0, 1.0]  # Sleep, Morning, Afternoon, Evening
                    
                    for _, row in user_data.iterrows():
                        rec_start = row['start_timestamp']
                        rec_end = row['end_timestamp']
                        
                        # Find the span of calendar days covered by the record
                        current_day = pd.Timestamp(rec_start.date())
                        last_day = pd.Timestamp(rec_end.date())
                        
                        total_weighted = 0.0
                        record_weighted_dict = {}
                        
                        day_iter = current_day
                        while day_iter <= last_day:
                            block_offsets_daily = [
                                (timedelta(hours=0), timedelta(hours=8)),
                                (timedelta(hours=8), timedelta(hours=12)),
                                (timedelta(hours=12), timedelta(hours=17)),
                                (timedelta(hours=17), timedelta(hours=24))
                            ]
                            
                            for i, (b_start, b_end) in enumerate(block_offsets_daily):
                                bs = day_iter + b_start
                                be = day_iter + b_end
                                
                                if rec_end > bs and rec_start < be:
                                    overlap_s = max(rec_start, bs)
                                    overlap_e = min(rec_end, be)
                                    dur = (overlap_e - overlap_s).total_seconds()
                                    weighted_dur = dur * weights[i]
                                    
                                    total_weighted += weighted_dur
                                    record_weighted_dict[(bs, be)] = weighted_dur
                                    
                            day_iter += timedelta(days=1)
                            
                        # Distribute based on proportion of total weighted duration
                        if total_weighted > 0:
                            for b_idx, (b_start, b_end) in enumerate(blocks):
                                if (b_start, b_end) in record_weighted_dict:
                                    proportion = record_weighted_dict[(b_start, b_end)] / total_weighted
                                    block_values[b_idx] += row[val_col] * proportion
                        else:
                            # Equal split fallback
                            overlapping_target_blocks = [
                                b_idx for b_idx, (bs, be) in enumerate(blocks) 
                                if rec_end > bs and rec_start < be
                            ]
                            if overlapping_target_blocks:
                                proportion = 1.0 / len(overlapping_target_blocks)
                                for b_idx in overlapping_target_blocks:
                                    block_values[b_idx] += row[val_col] * proportion
                else:
                    # Point logic fallback
                    for b_idx, (b_start, b_end) in enumerate(blocks):
                        overlap_mask = (user_data['start_timestamp'] >= b_start) & (user_data['start_timestamp'] < b_end)
                        block_values[b_idx] = user_data[overlap_mask][val_col].sum()
                        
            all_modality_features.append(block_values)
            
        # Return sequence vector
        return np.stack(all_modality_features, axis=-1)



