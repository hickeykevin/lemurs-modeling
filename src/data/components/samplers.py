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

class IntervalAwareSampler(TimeSampler):
    """Samples health metrics into survey-anchored, non-uniform, interval-aware bins.

    This sampler differs from ``RollingSampler``/``OffsetSampler`` in three ways,
    each addressing a specific property of the underlying HealthKit/Health Connect data.

    1. **Interval-aware allocation.** Health records are intervals, not points.
       iPhone step records in this dataset run ~273 minutes on average, so a single
       record routinely straddles several bins. Assigning a record's whole value to
       the bin containing its ``start_timestamp`` (what the floor-and-groupby
       samplers do) manufactures activity spikes at record-start times and
       artificial troughs after them — temporal structure that is an artifact of
       when the OS chose to close a record, not of behaviour. Here each record's
       value is split across bins in proportion to its temporal overlap with them.

       A record of 1,200 steps spanning 08:00-14:00, against bins [06:00-12:00) and
       [12:00-18:00), contributes 800 and 400 respectively rather than 1,200 and 0.

       Allocation is proportional to the fraction of the *record's own* duration
       falling in each bin, so any portion of a record extending outside the
       lookback window is dropped rather than redistributed inward.

    2. **Non-uniform bins.** ``bin_edges_hours`` are hours *before* the survey
       anchor. The default ``[0, 3, 6, 12, 24, 48, 72]`` yields six bins that are
       fine near the survey — where the label's "since last prompt" referent lies —
       and coarse further back, where the data serves as personal baseline context.
       This puts resolution where the label is while keeping input dimensionality
       low (6 steps vs. 18 for a uniform 6-hour grid over 3 days).

    3. **Mask channels.** Missingness in this dataset is block-structured by user:
       people who sync health data have it densely, people who do not have none for
       days at a stretch. Zero-filling an unobserved bin makes it indistinguishable
       from a genuinely sedentary one, which are opposite signals. When
       ``emit_mask`` is set, each modality gains a companion channel that is 1.0
       where at least one record overlapped the bin and 0.0 where none did.

    Because the bins have different widths, values are emitted as **rates**
    (value per hour) by default. Every bin shares a feature column in the returned
    ``[Time, Features]`` matrix, so a downstream scaler standardises a 3-hour bin
    and a 24-hour bin together; without rate normalisation that comparison is
    meaningless. Set ``normalize_by_duration=False`` to emit raw sums instead.

    Channel order is ``[mod_0_value, mod_0_mask, mod_1_value, mod_1_mask, ...]``
    followed by optional cyclic time features. Bins are returned chronologically
    (oldest first), so the final timestep is the interval ending at the survey.

    Args:
        bin_edges_hours: Ascending bin boundaries in hours before the survey
            timestamp. ``[0, 3, 6]`` means bins ``[t-3h, t)`` and ``[t-6h, t-3h)``.
        emit_mask: Whether to append a per-modality observed/missing channel.
        normalize_by_duration: Emit per-hour rates rather than raw bin sums.
        include_time_features: Append cyclic hour-of-day and weekday channels.
            Unlike ``OffsetSampler``, bins here are anchored to the survey
            timestamp rather than midnight, so bin-centre hours genuinely vary
            across samples and carry signal.

    Example:
        >>> sampler = IntervalAwareSampler(bin_edges_hours=[0, 3, 6, 12, 24, 48, 72])
        >>> features = sampler(survey_ts, user_id, dfs, cols, ['step', 'calorie'])
        >>> features.shape
        (6, 8)
    """

    def __init__(
        self,
        bin_edges_hours: List[float] = [0, 3, 6, 12, 24, 48, 72],
        emit_mask: bool = True,
        normalize_by_duration: bool = True,
        include_time_features: bool = True,
        **kwargs,
    ):
        edges = list(bin_edges_hours)
        if len(edges) < 2:
            raise ValueError(f"bin_edges_hours needs at least 2 edges; got {edges!r}")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"bin_edges_hours must be strictly ascending; got {edges!r}")
        if edges[0] < 0:
            raise ValueError(f"bin_edges_hours must be non-negative; got {edges!r}")

        self.bin_edges_hours = edges
        self.emit_mask = emit_mask
        self.normalize_by_duration = normalize_by_duration
        self.include_time_features = include_time_features

        # Bin i spans [anchor - edges[i+1], anchor - edges[i]), built oldest-first
        # so index 0 is the furthest back and the last index ends at the survey.
        self._offsets = [
            (edges[i + 1], edges[i]) for i in range(len(edges) - 1)
        ][::-1]
        self._durations_h = np.array(
            [start - end for start, end in self._offsets], dtype=np.float64
        )

    def __call__(self, survey_timestamp, app_user_id, modality_dfs, modality_cols, modalities):
        anchor = pd.Timestamp(survey_timestamp)
        bin_starts = [anchor - timedelta(hours=s) for s, _ in self._offsets]
        bin_ends = [anchor - timedelta(hours=e) for _, e in self._offsets]

        n_bins = len(self._offsets)
        window_start = bin_starts[0]
        window_end = bin_ends[-1]

        bin_starts_ns = np.array([b.value for b in bin_starts], dtype=np.int64)
        bin_ends_ns = np.array([b.value for b in bin_ends], dtype=np.int64)

        all_modality_features = []
        for mod in modalities:
            df = modality_dfs[mod]
            val_col = modality_cols[mod]

            values = np.zeros(n_bins, dtype=np.float32)
            observed = np.zeros(n_bins, dtype=np.float32)

            # Any record overlapping the window: ends after it starts and starts
            # before it ends. Point records (no end_timestamp) fall back to start.
            has_end = "end_timestamp" in df.columns
            end_series = df["end_timestamp"] if has_end else df["start_timestamp"]
            mask = (
                (df["app_user_id"] == app_user_id)
                & (end_series > window_start)
                & (df["start_timestamp"] < window_end)
            )
            user_data = df[mask]

            if not user_data.empty:
                rec_starts = user_data["start_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
                if has_end:
                    rec_ends = user_data["end_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
                else:
                    rec_ends = rec_starts
                rec_values = user_data[val_col].values.astype(np.float64)

                # Guard against corrupt rows where the interval runs backwards.
                valid = rec_ends >= rec_starts
                rec_starts, rec_ends, rec_values = rec_starts[valid], rec_ends[valid], rec_values[valid]

                if rec_starts.size:
                    # Overlap of every (record, bin) pair, in nanoseconds.
                    overlap_s = np.maximum(rec_starts[:, None], bin_starts_ns[None, :])
                    overlap_e = np.minimum(rec_ends[:, None], bin_ends_ns[None, :])
                    overlap = np.maximum(0, overlap_e - overlap_s).astype(np.float64)  # [R, B]

                    rec_durations = (rec_ends - rec_starts).astype(np.float64)  # [R]

                    # Instantaneous records carry no duration to split, so they are
                    # assigned wholly to the bin containing their timestamp.
                    is_point = rec_durations <= 0
                    proportions = np.zeros_like(overlap)

                    if (~is_point).any():
                        proportions[~is_point] = (
                            overlap[~is_point] / rec_durations[~is_point, None]
                        )
                    if is_point.any():
                        pt_starts = rec_starts[is_point]
                        in_bin = (
                            (pt_starts[:, None] >= bin_starts_ns[None, :])
                            & (pt_starts[:, None] < bin_ends_ns[None, :])
                        )
                        proportions[is_point] = in_bin.astype(np.float64)

                    values = (rec_values[:, None] * proportions).sum(axis=0).astype(np.float32)

                    # A bin counts as observed if any record covered part of it,
                    # regardless of whether the value it contributed was zero.
                    covered = (overlap > 0)
                    covered[is_point] = proportions[is_point] > 0
                    observed = covered.any(axis=0).astype(np.float32)

            if self.normalize_by_duration:
                values = (values / self._durations_h).astype(np.float32)

            all_modality_features.append(values)
            if self.emit_mask:
                all_modality_features.append(observed)

        if self.include_time_features:
            centers = [
                s + (e - s) / 2 for s, e in zip(bin_starts, bin_ends)
            ]
            hours = np.array([c.hour + c.minute / 60.0 for c in centers], dtype=np.float32)
            weekdays = np.array([c.weekday() for c in centers], dtype=np.float32)

            all_modality_features.append(np.sin(2 * np.pi * hours / 24.0).astype(np.float32))
            all_modality_features.append(np.cos(2 * np.pi * hours / 24.0).astype(np.float32))
            all_modality_features.append(np.sin(2 * np.pi * weekdays / 7.0).astype(np.float32))
            all_modality_features.append(np.cos(2 * np.pi * weekdays / 7.0).astype(np.float32))

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
