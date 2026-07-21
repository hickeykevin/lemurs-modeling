import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Literal
from src.utils.database_service import DatabaseService
from src.data.components.label_aggregators import LabelAggregator

class CohortBuilder:
    """Builds and filters the cohort dataset.

    This class coordinates database connections, applies data-cleaning
    modality preprocessors, performs global user/time filters, clips outliers,
    aggregates target daily clinical labels, and filters by device operating system (OS).
    """

    def __init__(
        self,
        modalities: List[str],
        modality_cols: Dict[str, str],
        preprocessors: Optional[Dict[str, Any]],
        aggregator: LabelAggregator,
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "mean",
        use_demographics: bool = False,
        use_sleep: bool = False,
        enrollment_lead_days: float = 7.0,
        enrollment_trail_days: float = 1.0,
    ):
        """Initializes the CohortBuilder.

        Args:
            modalities: List of database tables to fetch (e.g., ["step", "calorie"]).
            modality_cols: Map of modality names to their value column (e.g., {"step": "steps"}).
            preprocessors: Map of modality names to preprocessor callables.
            aggregator: Strategy object for binarizing/aggregating daily survey responses.
            os_filter: Operating system filter mode ("ios", "android", or "both").
            collapse_strategy: Aggregation strategy to collapse multiple daily survey responses for non-yes-no questions.
            use_demographics: Whether to load and merge demographics.
            use_sleep: Whether to load and compute sleep features.
            enrollment_lead_days: Days of sensor data to retain *before* a user's
                first survey. Must exceed the sampler's lookback so that early
                surveys still see a full window.
            enrollment_trail_days: Days of sensor data to retain after a user's
                last survey.
        """
        self.modalities = modalities
        self.modality_cols = modality_cols
        self.preprocessors = preprocessors
        self.aggregator = aggregator
        self.os_filter = os_filter
        self.collapse_strategy = collapse_strategy
        self.use_demographics = use_demographics
        self.use_sleep = use_sleep
        self.enrollment_lead_days = enrollment_lead_days
        self.enrollment_trail_days = enrollment_trail_days

    def build(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """Orchestrates the entire cohort building pipeline.

        This method connects to the database, extracts and cleans all requested
        health metrics, fetches and binarizes survey answers, extracts user demographics,
        and then filters the cohort to match OS-demographic criteria.

        Returns:
            A tuple containing:
                - Dict[str, pd.DataFrame]: Cleaned and preprocessed modality dataframes.
                - pd.DataFrame: Combined target survey labels (master dataframe).
                - pd.DataFrame: Cleaned demographics dataframe.
        """
        db = DatabaseService()
        if not db.connect():
            raise Exception("Failed to connect to the database.")

        try:
            # Step 1 & 2: Extract data streams, apply cleanings/aggregations, and extract demographics
            enrollment_windows = self._get_enrollment_windows(db)
            modality_dfs = self._extract_and_clean_modalities(db, enrollment_windows)
            master_df = self._extract_and_aggregate_labels(db)
            if self.use_demographics:
                demographics_df = self._extract_demographics(db)
            else:
                demographics_df = pd.DataFrame(columns=['app_user_id', 'gender', 'age', 'lgbt'])
        finally:
            # Ensure database connection is closed regardless of success/error
            db.disconnect()

        # Step 3: Filter cohort demographics by operating system if requested
        master_df = self._apply_os_filtering(master_df, modality_dfs)
        
        return modality_dfs, master_df, demographics_df

    def _extract_demographics(self, db: DatabaseService) -> pd.DataFrame:
        """Extracts, filters and pivots user demographic information.

        Args:
            db: Connected DatabaseService instance.

        Returns:
            pd.DataFrame: Cleaned and pivoted demographics dataframe.
        """
        df = db.extract_from_database("demographic")
        
        # Remove test and discontinued users
        drop_users = [1, 2, 3, 10, 21, 22, 43, 44]
        if 'app_user_id' in df.columns:
            df = df[~df['app_user_id'].isin(drop_users)]
            
        # Pivot EAV structure
        if not df.empty and 'keyword' in df.columns and 'value' in df.columns:
            # Pivot, index on app_user_id
            df_pivot = df.pivot(index='app_user_id', columns='keyword', values='value').reset_index()
            # Rename columns to standard names
            df_pivot = df_pivot.rename(columns={
                'gender identity': 'gender',
                'gender at birth': 'gender_at_birth'
            })
            # Ensure gender is there. Fallback to gender_at_birth if needed
            if 'gender' not in df_pivot.columns:
                df_pivot['gender'] = np.nan
            if 'gender_at_birth' in df_pivot.columns:
                df_pivot['gender'] = df_pivot['gender'].fillna(df_pivot['gender_at_birth'])
            
            return df_pivot
        else:
            return pd.DataFrame(columns=['app_user_id', 'gender', 'age', 'lgbt'])

    def _get_enrollment_windows(self, db: DatabaseService) -> pd.DataFrame:
        """Derives each user's active study window from their survey activity.

        Users enrolled on staggered dates spanning many months, so no single
        global cutoff separates study data from noise. Health apps also backfill
        historical data on first sync — some users carry records from years before
        enrollment, and occasional records land after their last survey. Both are
        outside the period the labels describe.

        Returns:
            pd.DataFrame: Columns ``app_user_id``, ``window_start``, ``window_end``.
        """
        sr = db.extract_from_database("survey_response")
        if sr.empty or "timestamp" not in sr.columns:
            return pd.DataFrame(columns=["app_user_id", "window_start", "window_end"])

        sr = sr.copy()
        sr["timestamp"] = pd.to_datetime(sr["timestamp"]).astype("datetime64[ns]")

        # Only the daily surveys (0=morning, 1=afternoon) bound the observation
        # period; the PHQ-9 (survey 2) is administered outside the daily cadence.
        if "survey_id" in sr.columns:
            daily = sr[sr["survey_id"].isin([0, 1])]
            if not daily.empty:
                sr = daily

        windows = sr.groupby("app_user_id")["timestamp"].agg(["min", "max"]).reset_index()
        windows["window_start"] = windows["min"] - pd.Timedelta(days=self.enrollment_lead_days)
        windows["window_end"] = windows["max"] + pd.Timedelta(days=self.enrollment_trail_days)
        return windows[["app_user_id", "window_start", "window_end"]]

    def _apply_enrollment_window(
        self, df: pd.DataFrame, enrollment_windows: pd.DataFrame
    ) -> pd.DataFrame:
        """Restricts sensor records to each user's own study window.

        Users with no survey activity have no window and are dropped entirely —
        they carry no labels, so their sensor data cannot contribute.
        """
        if df.empty or enrollment_windows.empty or "app_user_id" not in df.columns:
            return df

        merged = df.merge(enrollment_windows, on="app_user_id", how="left")
        in_window = (
            merged["window_start"].notna()
            & (merged["start_timestamp"] >= merged["window_start"])
            & (merged["start_timestamp"] <= merged["window_end"])
        )
        return merged.loc[in_window.values].drop(columns=["window_start", "window_end"])

    def _extract_and_clean_modalities(
        self, db: DatabaseService, enrollment_windows: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """Extracts, cleans, and applies outlier clipping to sensor modalities.

        Args:
            db: Connected DatabaseService instance.
            enrollment_windows: Per-user study windows used to drop backfilled and
                post-study sensor records. Falls back to a global cutoff if omitted.

        Returns:
            Dict[str, pd.DataFrame]: Map of cleaned modality dataframes.
        """
        modality_dfs = {}
        for mod in self.modalities:
            df = db.extract_from_database(mod)
            
            # 1. Apply modality-specific preprocessors if registered
            if self.preprocessors and mod in self.preprocessors:
                df = self.preprocessors[mod](df)
                
            # Coerce timestamps to standard datetime format
            df['start_timestamp'] = pd.to_datetime(df["start_timestamp"]).astype("datetime64[ns]")
            if "end_timestamp" in df.columns: 
                df["end_timestamp"] = pd.to_datetime(df["end_timestamp"]).astype("datetime64[ns]")
            
            # 2. Global Filters: Remove test and discontinued users
            # Test users: [1, 2, 3, 10, 43]
            # Discontinued/Anomalous users: [21, 22]
            drop_users = [1, 2, 3, 10, 21, 22, 43, 44]
            if 'app_user_id' in df.columns:
                df = df[~df['app_user_id'].isin(drop_users)]
                
            # 3. Time filter: restrict each user's records to their own study
            # window. A global cutoff cannot do this correctly — enrollment is
            # staggered across months, so the same date is mid-study for one user
            # and years of backfill for another.
            if enrollment_windows is not None and not enrollment_windows.empty:
                df = self._apply_enrollment_window(df, enrollment_windows)
            else:
                df = df[df['start_timestamp'] >= pd.Timestamp('2025-09-01')]

            modality_dfs[mod] = df

        # 3. Outlier Clipping: Clip extreme values at the 99th percentile globally
        for mod, df in modality_dfs.items():
            val_col = self.modality_cols.get(mod)
            if val_col and val_col in df.columns:
                limit = df[val_col].quantile(0.99)
                df[val_col] = df[val_col].clip(upper=limit)

        return modality_dfs

    def _compute_sleep_features(self, answer_df: pd.DataFrame) -> pd.DataFrame:
        """Computes sleep features (sleep hours and category) for each survey response.

        Args:
            answer_df: Dataframe of collapsed answers (contains 'survey_response_id', 'question_id', 'answer').

        Returns:
            pd.DataFrame: Dataframe with 'survey_response_id', 'sleep_hours', and 'sleep_category'.
        """
        asleep_id = 54
        awake_id = 55

        # Filter for sleep answers
        target_answers = answer_df[answer_df['question_id'].isin([asleep_id, awake_id])]
        if target_answers.empty:
            return pd.DataFrame(columns=['survey_response_id', 'sleep_hours', 'sleep_category'])

        # Pivot to long format
        pivoted = target_answers.pivot(index='survey_response_id', columns='question_id', values='answer')

        # Ensure both question columns exist in the pivoted dataframe
        for qid in [asleep_id, awake_id]:
            if qid not in pivoted.columns:
                pivoted[qid] = None

        def calc_hours(row) -> Optional[float]:
            asleep = row[asleep_id]
            awake = row[awake_id]

            if pd.isna(asleep) or pd.isna(awake):
                return None

            try:
                def to_hours(t_str) -> Optional[float]:
                    if isinstance(t_str, (int, float)):
                        return float(t_str)
                    if not isinstance(t_str, str):
                        return None
                    t_str = t_str.strip().upper()
                    dt = pd.to_datetime(t_str, format="%I:%M %p")
                    return dt.hour + dt.minute / 60.0

                asleep_h = to_hours(asleep)
                awake_h = to_hours(awake)

                if asleep_h is None or awake_h is None:
                    return None

                if awake_h >= asleep_h:
                    return awake_h - asleep_h
                else:
                    return (awake_h - asleep_h) + 24.0
            except Exception:
                return None

        sleep_hours = pivoted.apply(calc_hours, axis=1)

        def categorize(hours) -> Optional[int]:
            if pd.isna(hours) or hours is None:
                return None
            if hours < 5.0:
                return 0
            elif hours <= 10.0:
                return 1
            else:
                return 2

        labels = sleep_hours.apply(categorize)

        res = pd.DataFrame({
            'sleep_hours': sleep_hours,
            'sleep_category': labels
        }).reset_index()
        return res

    def _extract_and_aggregate_labels(self, db: DatabaseService) -> pd.DataFrame:
        """Fetches survey answers and aggregates them into daily clinical labels.

        Args:
            db: Connected DatabaseService instance.

        Returns:
            pd.DataFrame: Merged and binarized target labels dataframe.
        """
        # Fetch underlying answers and parent survey responses
        answer_df = db.extract_from_database("answer")
        survey_response_df = db.extract_from_database("survey_response")
        survey_response_df['timestamp'] = pd.to_datetime(survey_response_df["timestamp"]).astype("datetime64[ns]")

        # Filter out duplicate responses submitted in quick succession
        survey_response_df, answer_df = self._filter_duplicate_responses(survey_response_df, answer_df)

        # Collapse multiple survey responses per user per day
        survey_response_df, answer_df = self._collapse_daily_responses(survey_response_df, answer_df, db)

        # Extract only question IDs required by the chosen aggregator
        question_ids = self.aggregator.get_question_ids()
        target_answers = answer_df[answer_df['question_id'].isin(question_ids)].copy()

        # Coerce answer strings to numeric scores, mapping "yes" -> 1.0 and "no" -> 0.0, and dropping any nulls
        # BUT ONLY IF the aggregator does not require strings!
        if not getattr(self.aggregator, 'requires_strings', False):
            clean_answers = target_answers['answer'].astype(str).str.strip().str.lower()
            mapped_answers = clean_answers.map({'yes': 1.0, 'no': 0.0})
            target_answers['answer'] = mapped_answers.fillna(pd.to_numeric(target_answers['answer'], errors='coerce'))
            target_answers = target_answers.dropna(subset=['answer'])
        else:
            # Just drop nulls, keep strings
            target_answers = target_answers.dropna(subset=['answer'])

        # Aggregate daily scores via the configured binarization strategy
        aggregated_answers = self.aggregator(target_answers)

        # Merge with survey responses to retrieve app_user_id and target timestamp
        master_df = pd.merge(
            aggregated_answers,
            survey_response_df[['id', 'app_user_id', 'timestamp']],
            left_on='survey_response_id',
            right_on='id',
            suffixes=('', '_survey')
        ).rename(columns={'timestamp': 'record_timestamp'})

        # If use_sleep is enabled, compute and merge sleep features
        if getattr(self, 'use_sleep', False):
            sleep_df = self._compute_sleep_features(answer_df)
            master_df = pd.merge(master_df, sleep_df, on='survey_response_id', how='left')

        return master_df

    def _collapse_daily_responses(
        self, survey_response_df: pd.DataFrame, answer_df: pd.DataFrame, db: DatabaseService
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collapses multiple survey responses for the same user on the same calendar day into a single response.

        Yes-no questions are aggregated using 'max' (conservative).
        All other questions are aggregated using 'mean'.

        Args:
            survey_response_df: Dataframe of survey responses.
            answer_df: Dataframe of answers.
            db: Connected DatabaseService instance.

        Returns:
            Tuple of collapsed (survey_response_df, answer_df).
        """
        if self.collapse_strategy == "none" or self.collapse_strategy is None:
            return survey_response_df, answer_df

        if survey_response_df.empty or answer_df.empty:
            return survey_response_df, answer_df

        # Get question styles to identify yes-no questions
        try:
            question_df = db.extract_from_database("question")
            yes_no_qids = set(question_df[question_df['style'] == 'yes-no']['id'])
        except Exception:
            # Fallback to standard yes-no question IDs from the schema if query fails
            yes_no_qids = {8, 11, 12, 13, 15, 16, 17, 18}

        # Work on copies
        sr = survey_response_df.copy()
        sr['timestamp'] = pd.to_datetime(sr['timestamp']).astype('datetime64[ns]')
        sr['date'] = sr['timestamp'].dt.date

        # Clean/coerce answers to numeric
        ans = answer_df.copy()
        
        # Identify string-based questions needed by the aggregator
        string_qids = set()
        if getattr(self.aggregator, 'requires_strings', False):
            string_qids = set(self.aggregator.get_question_ids())
        if getattr(self, 'use_sleep', False):
            string_qids.update({54, 55})

        # Split into string-based and numeric-based answers
        is_str_q = ans['question_id'].isin(string_qids)
        ans_str = ans[is_str_q].copy()
        ans_num = ans[~is_str_q].copy()

        # Process numeric answers as before
        ans_num['answer_clean'] = ans_num['answer'].astype(str).str.strip().str.lower()
        mapped_vals = ans_num['answer_clean'].map({'yes': 1.0, 'no': 0.0})
        ans_num['answer_numeric'] = mapped_vals.fillna(pd.to_numeric(ans_num['answer_clean'], errors='coerce'))
        ans_num = ans_num.dropna(subset=['answer_numeric'])

        # For string answers, we keep the original string value in 'answer_numeric'
        ans_str['answer_numeric'] = ans_str['answer']
        ans_str = ans_str.dropna(subset=['answer_numeric'])

        # Combine back
        ans_processed = pd.concat([ans_num, ans_str], ignore_index=True)

        # Merge answers with survey response to obtain app_user_id and date
        merged = pd.merge(ans_processed, sr[['id', 'app_user_id', 'date']], left_on='survey_response_id', right_on='id')

        # Find the latest survey response ID per user-date to act as the daily representative
        latest_sr = sr.sort_values('timestamp').groupby(['app_user_id', 'date']).last().reset_index()

        # Group by app_user_id, date, question_id and aggregate
        is_yes_no = merged['question_id'].isin(yes_no_qids)
        is_string = merged['question_id'].isin(string_qids)
        
        merged_yn = merged[is_yes_no]
        merged_other = merged[~is_yes_no & ~is_string]
        merged_str = merged[is_string]
        
        agg_yn = merged_yn.groupby(['app_user_id', 'date', 'question_id'])['answer_numeric'].max().reset_index() if not merged_yn.empty else pd.DataFrame(columns=['app_user_id', 'date', 'question_id', 'answer_numeric'])
        agg_other = merged_other.groupby(['app_user_id', 'date', 'question_id'])['answer_numeric'].agg(self.collapse_strategy).reset_index() if not merged_other.empty else pd.DataFrame(columns=['app_user_id', 'date', 'question_id', 'answer_numeric'])
        agg_str = merged_str.groupby(['app_user_id', 'date', 'question_id'])['answer_numeric'].last().reset_index() if not merged_str.empty else pd.DataFrame(columns=['app_user_id', 'date', 'question_id', 'answer_numeric'])
        
        collapsed_ans = pd.concat([agg_yn, agg_other, agg_str], ignore_index=True)
        
        # Link daily aggregated answers to the representative survey_response_id
        collapsed_ans = pd.merge(
            collapsed_ans,
            latest_sr[['app_user_id', 'date', 'id']],
            on=['app_user_id', 'date']
        ).rename(columns={'id': 'survey_response_id', 'answer_numeric': 'answer'})

        # Keep only the representative survey responses in the main dataframe
        collapsed_sr = latest_sr.drop(columns=['date'])
        
        # Recreate sequential id column
        collapsed_ans['id'] = range(len(collapsed_ans))
        collapsed_ans = collapsed_ans[['id', 'survey_response_id', 'question_id', 'answer']]
        
        # Log summary
        orig_count = len(survey_response_df)
        new_count = len(collapsed_sr)
        if orig_count > new_count:
            import logging
            logging.getLogger(__name__).warning(
                f"Collapsed {orig_count} survey responses into {new_count} unique daily samples per user."
            )

        return collapsed_sr, collapsed_ans

    def _filter_duplicate_responses(
        self, survey_response_df: pd.DataFrame, answer_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filters out duplicate survey responses submitted in quick succession (within 10 minutes)
        by the same user for the same survey type with identical answers.

        Args:
            survey_response_df: Dataframe of survey responses.
            answer_df: Dataframe of answers.

        Returns:
            Tuple of filtered (survey_response_df, answer_df).
        """
        if survey_response_df.empty or answer_df.empty:
            return survey_response_df, answer_df

        # Ensure timestamps are datetime
        sr = survey_response_df.copy()
        sr['timestamp'] = pd.to_datetime(sr['timestamp']).astype('datetime64[ns]')
        
        # Determine sorting and grouping keys (survey_id is optional to support mock schemas)
        groupby_cols = ['app_user_id']
        if 'survey_id' in sr.columns:
            groupby_cols.append('survey_id')
            
        duplicate_ids = set()
        
        # Group answers by survey_response_id for O(1) comparison
        import collections
        ans_by_sr = collections.defaultdict(dict)
        for _, row in answer_df.iterrows():
            sr_id = row['survey_response_id']
            q_id = row['question_id']
            val = row['answer']
            ans_by_sr[sr_id][q_id] = val
            
        for _, group in sr.groupby(groupby_cols):
            # Sort responses chronologically
            sorted_group = group.sort_values('timestamp')
            
            last_kept_id = None
            last_kept_time = None
            
            for _, row in sorted_group.iterrows():
                curr_id = row['id']
                curr_time = row['timestamp']
                
                if last_kept_id is None:
                    last_kept_id = curr_id
                    last_kept_time = curr_time
                    continue
                    
                time_diff = curr_time - last_kept_time
                
                if time_diff <= pd.Timedelta(minutes=10):
                    ans_prev = ans_by_sr.get(last_kept_id, {})
                    ans_curr = ans_by_sr.get(curr_id, {})
                    
                    if ans_prev == ans_curr:
                        duplicate_ids.add(curr_id)
                        continue
                        
                # Not a duplicate, update reference points
                last_kept_id = curr_id
                last_kept_time = curr_time
                
        if duplicate_ids:
            import logging
            logging.getLogger(__name__).warning(
                f"Filtering out {len(duplicate_ids)} duplicate survey responses submitted within 10 minutes with identical answers."
            )
            survey_response_df = survey_response_df[~survey_response_df['id'].isin(duplicate_ids)]
            answer_df = answer_df[~answer_df['survey_response_id'].isin(duplicate_ids)]
            
        return survey_response_df, answer_df

    def _apply_os_filtering(self, master_df: pd.DataFrame, modality_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Filters target labels to keep only users on the specified operating system.

        Determines a user's operating system dynamically by looking at their 
        active `app_source` strings across all modality datasets.

        Args:
            master_df: Target clinical labels dataframe.
            modality_dfs: Cleaned modality dataframes containing sensor data.

        Returns:
            pd.DataFrame: OS-filtered master dataframe.
        """
        if not self.os_filter or self.os_filter == "both" or not modality_dfs:
            return master_df
            
        # Group all unique app sources encountered per user
        user_sources = {}
        for df in modality_dfs.values():
            if 'app_user_id' in df.columns and 'app_source' in df.columns:
                for user_id, group in df.groupby('app_user_id'):
                    sources = group['app_source'].dropna().unique()
                    if user_id not in user_sources:
                        user_sources[user_id] = set()
                    user_sources[user_id].update(str(s) for s in sources)
                    
        # Determine OS mapping (ios, android, or both) per user ID
        user_os = {}
        for user_id, sources in user_sources.items():
            is_ios = any('iPhone' in s or 'Apple Watch' in s or 'HealthKit' in s for s in sources)
            is_android = any('androidx' in s for s in sources)
            user_os[user_id] = 'ios' if is_ios and not is_android else ('android' if is_android and not is_ios else 'both')
        
        # Filter the cohort according to the target OS filter configuration
        target_os = self.os_filter.lower()
        allowed_users = [u for u, os_typ in user_os.items() if os_typ == target_os]
        return master_df[master_df['app_user_id'].isin(allowed_users)]
