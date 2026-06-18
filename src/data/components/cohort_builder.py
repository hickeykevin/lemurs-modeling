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
        os_filter: Optional[Literal["ios", "android", "both"]] = "both"
    ):
        """Initializes the CohortBuilder.

        Args:
            modalities: List of database tables to fetch (e.g., ["step", "calorie"]).
            modality_cols: Map of modality names to their value column (e.g., {"step": "steps"}).
            preprocessors: Map of modality names to preprocessor callables.
            aggregator: Strategy object for binarizing/aggregating daily survey responses.
            os_filter: Operating system filter mode ("ios", "android", or "both").
        """
        self.modalities = modalities
        self.modality_cols = modality_cols
        self.preprocessors = preprocessors
        self.aggregator = aggregator
        self.os_filter = os_filter

    def build(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Orchestrates the entire cohort building pipeline.

        This method connects to the database, extracts and cleans all requested
        health metrics, fetches and binarizes survey answers, and then filters
        the cohort to match OS-demographic criteria.

        Returns:
            A tuple containing:
                - Dict[str, pd.DataFrame]: Cleaned and preprocessed modality dataframes.
                - pd.DataFrame: Combined target survey labels (master dataframe).
        """
        db = DatabaseService()
        if not db.connect():
            raise Exception("Failed to connect to the database.")

        try:
            # Step 1 & 2: Extract data streams and apply cleanings/aggregations
            modality_dfs = self._extract_and_clean_modalities(db)
            master_df = self._extract_and_aggregate_labels(db)
        finally:
            # Ensure database connection is closed regardless of success/error
            db.disconnect()

        # Step 3: Filter cohort demographics by operating system if requested
        master_df = self._apply_os_filtering(master_df, modality_dfs)
        
        return modality_dfs, master_df

    def _extract_and_clean_modalities(self, db: DatabaseService) -> Dict[str, pd.DataFrame]:
        """Extracts, cleans, and applies outlier clipping to sensor modalities.

        Args:
            db: Connected DatabaseService instance.

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
                
            # Time filter: drop all sensor data before 2025-09-01
            df = df[df['start_timestamp'] >= pd.Timestamp('2025-09-01')]
            
            modality_dfs[mod] = df

        # 3. Outlier Clipping: Clip extreme values at the 99th percentile globally
        for mod, df in modality_dfs.items():
            val_col = self.modality_cols.get(mod)
            if val_col and val_col in df.columns:
                limit = df[val_col].quantile(0.99)
                df[val_col] = df[val_col].clip(upper=limit)

        return modality_dfs

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

        # Extract only question IDs required by the chosen aggregator
        question_ids = self.aggregator.get_question_ids()
        target_answers = answer_df[answer_df['question_id'].isin(question_ids)].copy()

        # Coerce answer strings to numeric scores, mapping "yes" -> 1.0 and "no" -> 0.0, and dropping any nulls
        clean_answers = target_answers['answer'].astype(str).str.strip().str.lower()
        mapped_answers = clean_answers.map({'yes': 1.0, 'no': 0.0})
        target_answers['answer'] = mapped_answers.fillna(pd.to_numeric(target_answers['answer'], errors='coerce'))
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
        
        return master_df

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
