import pandas as pd
from typing import Optional

class ModalityPreprocessor:
    """Base class for modality-specific data preprocessing."""
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class StepPreprocessor(ModalityPreprocessor):
    """Data cleaning pipeline for step counts."""
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert to datetime temporarily if not already
        start_dt = pd.to_datetime(df["start_timestamp"])
        end_dt = pd.to_datetime(df["end_timestamp"]) if "end_timestamp" in df.columns else None

        # 1. Remove 0 duration records with steps > 0
        if end_dt is not None and "steps" in df.columns:
            duration = (end_dt - start_dt).dt.total_seconds() / 60.0
            zero_duration_mask = (duration == 0) & (df["steps"] > 0)
            df = df[~zero_duration_mask]
                
        # 2. Drop duplicates
        subset_cols = ["app_user_id", "start_timestamp", "end_timestamp", "steps", "app_source"]
        subset_cols = [c for c in subset_cols if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="first")
            
        # 3. Rename app_source 'Health' to 'Apple Watch' for app_user_id 26
        if "app_source" in df.columns and "app_user_id" in df.columns:
            mask = (df["app_user_id"] == 26) & (df["app_source"] == "Health")
            df.loc[mask, "app_source"] = "Apple Watch"

            # app_user_id=22 has some android records, but is iphone user, remove android records
            mask = (df["app_user_id"] == 22) & (df["app_source"].str.contains("android", case=False, na=False))
            df = df[~mask]

            
        return df
