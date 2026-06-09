import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Literal
from src.utils.database_service import DatabaseService
from src.data.components.label_aggregators import LabelAggregator

class CohortBuilder:
    """Builds the cohort dataset by extracting from the DB, applying preprocessors,
    and handling demographic (OS) filtering and target aggregations.
    """
    def __init__(
        self,
        modalities: List[str],
        modality_cols: Dict[str, str],
        preprocessors: Optional[Dict[str, Any]],
        aggregator: LabelAggregator,
        os_filter: Optional[Literal["ios", "android", "both"]] = "both"
    ):
        self.modalities = modalities
        self.modality_cols = modality_cols
        self.preprocessors = preprocessors
        self.aggregator = aggregator
        self.os_filter = os_filter

    def build(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Connects to the database and orchestrates the dataset build."""
        db = DatabaseService()
        if not db.connect():
            raise Exception("Failed to connect to the database.")

        try:
            modality_dfs = self._extract_and_clean_modalities(db)
            master_df = self._extract_and_aggregate_labels(db)
        finally:
            db.disconnect()

        # Apply OS filter
        master_df = self._apply_os_filtering(master_df, modality_dfs)
        
        return modality_dfs, master_df

    def _extract_and_clean_modalities(self, db: DatabaseService) -> Dict[str, pd.DataFrame]:
        modality_dfs = {}
        for mod in self.modalities:
            df = db.extract_from_database(mod)
            
            # 1. Apply modality-specific preprocessors
            if self.preprocessors and mod in self.preprocessors:
                df = self.preprocessors[mod](df)
                
            df['start_timestamp'] = pd.to_datetime(df["start_timestamp"]).astype("datetime64[ns]")
            if "end_timestamp" in df.columns: 
                df["end_timestamp"] = pd.to_datetime(df["end_timestamp"]).astype("datetime64[ns]")
            
            # 2. Global Filters
            # Remove test users and discontinued users
            drop_users = [1, 2, 3, 10, 21, 22, 43]
            if 'app_user_id' in df.columns:
                df = df[~df['app_user_id'].isin(drop_users)]
                
            # Remove records before 2025-01-01
            df = df[df['start_timestamp'] >= pd.Timestamp('2025-01-01')]
            
            modality_dfs[mod] = df

        # 3. Clip extreme outliers at 99th percentile globally
        for mod, df in modality_dfs.items():
            val_col = self.modality_cols.get(mod)
            if val_col and val_col in df.columns:
                limit = df[val_col].quantile(0.99)
                df[val_col] = df[val_col].clip(upper=limit)

        return modality_dfs

    def _extract_and_aggregate_labels(self, db: DatabaseService) -> pd.DataFrame:
        answer_df = db.extract_from_database("answer")
        survey_response_df = db.extract_from_database("survey_response")
        survey_response_df['timestamp'] = pd.to_datetime(survey_response_df["timestamp"]).astype("datetime64[ns]")

        question_ids = self.aggregator.get_question_ids()
        target_answers = answer_df[answer_df['question_id'].isin(question_ids)].copy()

        target_answers['answer'] = pd.to_numeric(target_answers['answer'], errors='coerce')
        target_answers = target_answers.dropna(subset=['answer'])

        aggregated_answers = self.aggregator(target_answers)

        master_df = pd.merge(
            aggregated_answers,
            survey_response_df[['id', 'app_user_id', 'timestamp']],
            left_on='survey_response_id',
            right_on='id',
            suffixes=('', '_survey')
        ).rename(columns={'timestamp': 'record_timestamp'})
        
        return master_df

    def _apply_os_filtering(self, master_df: pd.DataFrame, modality_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not self.os_filter or self.os_filter == "both" or not modality_dfs:
            return master_df
            
        user_sources = {}
        for df in modality_dfs.values():
            if 'app_user_id' in df.columns and 'app_source' in df.columns:
                for user_id, group in df.groupby('app_user_id'):
                    sources = group['app_source'].dropna().unique()
                    if user_id not in user_sources:
                        user_sources[user_id] = set()
                    user_sources[user_id].update(str(s) for s in sources)
                    
        user_os = {}
        for user_id, sources in user_sources.items():
            is_ios = any('iPhone' in s or 'Apple Watch' in s or 'HealthKit' in s for s in sources)
            is_android = any('androidx' in s for s in sources)
            user_os[user_id] = 'ios' if is_ios and not is_android else ('android' if is_android and not is_ios else 'both')
        
        target_os = self.os_filter.lower()
        allowed_users = [u for u, os_typ in user_os.items() if os_typ == target_os]
        return master_df[master_df['app_user_id'].isin(allowed_users)]
