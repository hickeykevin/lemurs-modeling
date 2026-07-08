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
    
    def __init__(self, lookback_hours: float = 24.0, resample_freq: str = "1h", include_time_features: bool = True, **kwargs):
        self.lookback_hours = lookback_hours
        self.resample_freq = resample_freq
        self.include_time_features = include_time_features
        
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

    
    def __init__(self, lookback_days: int = 1, include_time_features: bool = True, **kwargs):
        self.lookback_days = lookback_days
        self.include_time_features = include_time_features
        
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
                   
            user_data = df[mask]
            
            block_values = np.zeros(len(blocks), dtype=np.float32)
            
            if not user_data.empty:
                if 'end_timestamp' in user_data.columns:
                    # Probabilistic weight to prevent over-allocating activity to sleep blocks.
                    # Tiles across all days so shape matches [n_blocks].
                    block_weights = np.tile([0.05, 1.0, 1.0, 1.0], self.lookback_days)  # [B]

                    # Convert block and record boundaries to int64 nanoseconds for fast
                    # arithmetic without any Python-level datetime arithmetic in the loop.
                    block_starts_ns = np.array([b[0].value for b in blocks], dtype=np.int64)  # [B]
                    block_ends_ns   = np.array([b[1].value for b in blocks], dtype=np.int64)  # [B]

                    rec_starts_ns = user_data['start_timestamp'].values.astype('datetime64[ns]').astype(np.int64)  # [R]
                    rec_ends_ns   = user_data['end_timestamp'].values.astype('datetime64[ns]').astype(np.int64)    # [R]
                    rec_values    = user_data[val_col].values.astype(np.float64)           # [R]

                    # Broadcast to [R, B]: overlap start/end per (record, block) pair.
                    overlap_s = np.maximum(rec_starts_ns[:, None], block_starts_ns[None, :])
                    overlap_e = np.minimum(rec_ends_ns[:, None],   block_ends_ns[None, :])

                    # Duration in seconds; negative values mean no overlap → clamp to 0.
                    durations = np.maximum(0.0, (overlap_e - overlap_s) / 1e9)  # [R, B]

                    # Apply per-block weights and sum across blocks to get each record's
                    # total weighted coverage for normalisation.
                    weighted_durations = durations * block_weights[None, :]  # [R, B]
                    total_weighted = weighted_durations.sum(axis=1)           # [R]

                    # Proportions: how much of each record's value lands in each block.
                    with np.errstate(invalid='ignore', divide='ignore'):
                        proportions = np.where(
                            total_weighted[:, None] > 0,
                            weighted_durations / total_weighted[:, None],
                            0.0,
                        )  # [R, B]

                    # Fallback for records with zero total weighted duration (e.g. a record
                    # that sits entirely outside every block): split equally across any
                    # block that has raw (unweighted) overlap.
                    zero_total_mask = (total_weighted == 0)
                    if zero_total_mask.any():
                        has_overlap = (durations[zero_total_mask] > 0)          # [R_zero, B]
                        n_overlapping = has_overlap.sum(axis=1, keepdims=True).clip(min=1)
                        proportions[zero_total_mask] = has_overlap / n_overlapping

                    # Accumulate: for each block, sum value * proportion across all records.
                    block_values = (rec_values[:, None] * proportions).sum(axis=0).astype(np.float32)

                else:
                    # Point-in-time fallback when no end_timestamp column is present.
                    for b_idx, (b_start, b_end) in enumerate(blocks):
                        overlap_mask = (user_data['start_timestamp'] >= b_start) & (user_data['start_timestamp'] < b_end)
                        block_values[b_idx] = user_data[overlap_mask][val_col].sum()
                        
            all_modality_features.append(block_values)
            
        # Compute cyclic time features for each block in BlockSampler
        if self.include_time_features:
            block_centers = [b_start + (b_end - b_start) / 2.0 for b_start, b_end in blocks]
            hours = np.array([dt.hour for dt in block_centers], dtype=np.float32)
            sin_hour = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
            cos_hour = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
            
            weekdays = np.array([dt.weekday() for dt in block_centers], dtype=np.float32)
            sin_weekday = np.sin(2 * np.pi * weekdays / 7.0).astype(np.float32)
            cos_weekday = np.cos(2 * np.pi * weekdays / 7.0).astype(np.float32)
            
            all_modality_features.append(sin_hour)
            all_modality_features.append(cos_hour)
            all_modality_features.append(sin_weekday)
            all_modality_features.append(cos_weekday)
            
        # Return sequence vector
        return np.stack(all_modality_features, axis=-1)

class LagSampler(TimeSampler):
    """Sampler for the Lag-1 baseline.
    
    Instead of sampling health metrics, this sampler looks up the label of the 
    user's previous survey response. This allows the model to use the previous 
    state as the prediction for the current state.
    """
    def __init__(self, **kwargs):
        self.label_lookup = {} # (user_id, timestamp) -> prev_label

    def set_labels(self, master_df: pd.DataFrame):
        """Pre-computes the previous label for every record in the dataset."""
        # Ensure data is sorted by user and time
        df = master_df.sort_values(['app_user_id', 'record_timestamp'])
        
        # Shift labels by 1 within each user group
        df['prev_answer'] = df.groupby('app_user_id')['answer'].shift(1)
        
        # Fill missing previous labels (first sample for a user) with -1
        df['prev_answer'] = df['prev_answer'].fillna(-1.0)
        
        # Build lookup dictionary using a string key for robustness
        for _, row in df.iterrows():
            ts_str = row['record_timestamp'].isoformat()
            key = (int(row['app_user_id']), ts_str)
            self.label_lookup[key] = float(row['prev_answer'])

    def __call__(self, survey_timestamp, app_user_id, modality_dfs, modality_cols, modalities):
        # Lookup using the same string key format
        ts_str = survey_timestamp.isoformat()
        prev_label = self.label_lookup.get((int(app_user_id), ts_str), -1.0)
        
        # Return as [1, 1] feature matrix
        return np.array([[prev_label]], dtype=np.float32)
