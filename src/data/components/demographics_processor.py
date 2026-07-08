from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd


class DemographicsProcessor:
    """Extracts device source and preprocesses user demographics to construct feature maps."""

    def __init__(self, use_demographics: bool = True) -> None:
        self.use_demographics = use_demographics
        self.age_median = 30.0
        self.age_mean = 30.0
        self.age_std = 10.0
        self.gender_categories: List[str] = []
        self.lgbt_categories: List[str] = []
        self.demographics_dim = 4
        self.demographics_map: Dict[Any, np.ndarray] = {}
        self.default_demographics: np.ndarray = np.array([], dtype=np.float32)

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        demographics_df: pd.DataFrame,
        modality_dfs: Dict[str, pd.DataFrame],
        master_df: pd.DataFrame,
    ) -> Tuple[Dict[Any, np.ndarray], np.ndarray]:
        """Fits preprocessing parameters on the training set and transforms demographic records.

        Args:
            train_df: Training set labels dataframe.
            demographics_df: Raw demographic records dataframe.
            modality_dfs: Cleaned sensor modalities.
            master_df: The master linked labels dataframe (contains all users).

        Returns:
            Tuple containing:
                - demographics_map: Mapping of user IDs to embedding feature vectors.
                - default_demographics: Default vector for unseen or missing users.
        """
        # 1. Compute app_source context features for all users
        # Sources categories: [android, iphone, apple_watch, unknown]
        user_sources = {}
        if "step" in modality_dfs:
            step_df = modality_dfs["step"]
            if "app_user_id" in step_df.columns and "app_source" in step_df.columns:
                for uid, group in step_df.groupby("app_user_id"):
                    sources = group["app_source"].dropna().unique()

                    is_apple_watch = any("Apple Watch" in str(s) for s in sources)
                    is_iphone = any("Iphone" in str(s) or "iPhone" in str(s) for s in sources)
                    is_android = any("androidx" in str(s) or "health.connect" in str(s) for s in sources)

                    # Set precedence: Apple Watch > iPhone > Android
                    if is_apple_watch:
                        user_sources[uid] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                    elif is_iphone:
                        user_sources[uid] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                    elif is_android:
                        user_sources[uid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        user_sources[uid] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Fallback source vector for missing/unseen users
        default_source = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if self.use_demographics:
            # Preprocess demographics
            demographics_df = demographics_df.copy()
            for col in ["gender", "age", "lgbt"]:
                if col not in demographics_df.columns:
                    demographics_df[col] = np.nan

            train_users = train_df["app_user_id"].unique()
            train_demo = demographics_df[demographics_df["app_user_id"].isin(train_users)]

            # Fit age median and scale parameters
            age_train = pd.to_numeric(train_demo["age"], errors="coerce").dropna()
            self.age_median = age_train.median() if not age_train.empty else 30.0
            self.age_mean = age_train.mean() if not age_train.empty else 30.0
            self.age_std = age_train.std() if not age_train.empty and age_train.std() > 0 else 10.0

            # Get unique categories from training set
            self.gender_categories = [
                str(x).strip().lower() for x in train_demo["gender"].dropna().unique() if str(x).strip() != ""
            ]
            if not self.gender_categories:
                self.gender_categories = ["male", "female"]
            self.lgbt_categories = [
                str(x).strip().lower() for x in train_demo["lgbt"].dropna().unique() if str(x).strip() != ""
            ]
            if not self.lgbt_categories:
                self.lgbt_categories = ["no", "yes"]

            # Standardize age
            demographics_df["age"] = (
                pd.to_numeric(demographics_df["age"], errors="coerce").fillna(self.age_median)
            )
            demographics_df["age_scaled"] = (demographics_df["age"] - self.age_mean) / self.age_std

            # One-hot encode categoricals manually to ensure consistency
            for cat in self.gender_categories:
                demographics_df[f"gender_{cat}"] = (
                    demographics_df["gender"].astype(str).str.strip().str.lower() == cat
                ).astype(float)
            demographics_df["gender_unknown"] = (
                ~demographics_df["gender"].astype(str).str.strip().str.lower().isin(self.gender_categories)
            ).astype(float)

            for cat in self.lgbt_categories:
                demographics_df[f"lgbt_{cat}"] = (
                    demographics_df["lgbt"].astype(str).str.strip().str.lower() == cat
                ).astype(float)
            demographics_df["lgbt_unknown"] = (
                ~demographics_df["lgbt"].astype(str).str.strip().str.lower().isin(self.lgbt_categories)
            ).astype(float)

            # Processed feature list
            demographic_feature_cols = (
                ["age_scaled"]
                + [f"gender_{cat}" for cat in self.gender_categories]
                + ["gender_unknown"]
                + [f"lgbt_{cat}" for cat in self.lgbt_categories]
                + ["lgbt_unknown"]
            )
            self.demographics_dim = len(demographic_feature_cols) + 4

            # Map app_user_id to feature vector
            self.demographics_map = {}
            for _, row in demographics_df.iterrows():
                uid = row["app_user_id"]
                demo_feats = row[demographic_feature_cols].values.astype(np.float32)
                source_feats = user_sources.get(uid, default_source)
                self.demographics_map[uid] = np.concatenate([demo_feats, source_feats]).astype(np.float32)

            # Default demographic vector for missing/unseen users
            self.default_demographics = np.zeros(self.demographics_dim, dtype=np.float32)
            for idx, col in enumerate(demographic_feature_cols):
                if col in ["gender_unknown", "lgbt_unknown"]:
                    self.default_demographics[idx] = 1.0
            self.default_demographics[-4:] = default_source
        else:
            self.demographics_dim = 4
            self.default_demographics = default_source
            self.demographics_map = {}
            unique_users = master_df["app_user_id"].unique()
            for uid in unique_users:
                self.demographics_map[uid] = user_sources.get(uid, default_source)

        return self.demographics_map, self.default_demographics
