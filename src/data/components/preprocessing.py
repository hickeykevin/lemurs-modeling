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


class CaloriePreprocessor(ModalityPreprocessor):
    """Data cleaning pipeline for calorie records."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Convert start timestamp to datetime
        start_dt = pd.to_datetime(df["start_timestamp"])
        
        # 1. Remove records with identical start/end times if end_timestamp is present
        if "end_timestamp" in df.columns:
            end_dt = pd.to_datetime(df["end_timestamp"])
            same_time = start_dt == end_dt
            df = df.loc[~same_time].copy()
            # Update start_dt after filtering
            start_dt = pd.to_datetime(df["start_timestamp"])

        # 2. Remove records before study start (September 2025)
        study_start = pd.Timestamp("2025-09-01")
        df = df.loc[start_dt >= study_start].copy()

        # 3. Remove duplicate records
        subset_cols = [
            "app_user_id",
            "start_timestamp",
            "end_timestamp",
            "calories",
            "app_source",
        ]
        subset_cols = [c for c in subset_cols if c in df.columns]

        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="first")

        # 4. Android calorie unit correction
        # For any android user with calorie recorded above or equal to 3000, divide by 1000 and round to nearest integer
        if "app_source" in df.columns and "calories" in df.columns:
            android_mask = (
                df["app_source"].str.contains("android", case=False, na=False)
                & (df["calories"] >= 3000)
            )
            df.loc[android_mask, "calories"] = (
                df.loc[android_mask, "calories"] / 1000
            ).round().astype(int)

        return df


class DistancePreprocessor(ModalityPreprocessor):
    """Data cleaning pipeline for distance records."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Convert start timestamp to datetime
        start_dt = pd.to_datetime(df["start_timestamp"])

        # 1. Remove records with identical start/end timestamps if present
        if "end_timestamp" in df.columns:
            end_dt = pd.to_datetime(df["end_timestamp"])
            same_time = start_dt == end_dt
            df = df.loc[~same_time].copy()

            # Update start_dt after filtering
            start_dt = pd.to_datetime(df["start_timestamp"])

        # 2. Remove records before study start (September 2025)
        study_start = pd.Timestamp("2025-09-01")
        df = df.loc[start_dt >= study_start].copy()

        # 3. Remove duplicate records
        subset_cols = [
            "app_user_id",
            "start_timestamp",
            "end_timestamp",
            "distance",
            "app_source",
        ]
        subset_cols = [c for c in subset_cols if c in df.columns]

        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="first")

        return df
