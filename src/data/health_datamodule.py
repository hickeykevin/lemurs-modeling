from typing import Any, Dict, Optional, Tuple, List, Literal
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.database_service import DatabaseService
from src.data.components.health_dataset import HealthDataset
from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
from src.data.components.cohort_builder import CohortBuilder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.components.cohort_splitter import CohortSplitter
from src.data.components.demographics_processor import DemographicsProcessor
from src.data.components.prev_prediction_linker import PrevPredictionLinker

class HealthDataModule(LightningDataModule):
    """A DataModule for Multimodal longitudinal health data.

    This module handles fetching data from multiple database tables, linking them to 
    daily survey responses, and splitting the resulting dataset into training, 
    validation, and test sets using various strategies (random, user-level, or longitudinal).

    The module produces a `HealthDataset` which yields:
        - Features: [Time=24, Features=N_modalities] (Hourly resampled metrics)
        - Target: Integer survey answer

    Attributes:
        data_train: Training dataset.
        data_val: Validation dataset.
        data_test: Test dataset.
    """

    _MODALITY_MAPPING = {
        "step": "steps",
        "speed": "speed",
        "proximity": "number_of_devices",
        "distance": "distance",
        "calorie": "calories",
    }

    def __init__(
        self,
        aggregator: LabelAggregator,
        sampler: TimeSampler,
        scaler: Optional[Any] = None,
        preprocessors: Optional[Dict[str, Any]] = None,
        modalities: List[str] = ["step"],
        batch_size: int = 8,

        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        split_mode: Literal["user", "longitudinal"] = "user",
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "mean",
        use_demographics: bool = True,
        use_sleep: bool = False,
        enrollment_lead_days: float = 7.0,
        enrollment_trail_days: float = 1.0,
        require_sensor_data: bool = True,
        use_survey_context: bool = True,
        include_time_features: Optional[bool] = None,
        exclude_user_ids: Optional[List[int]] = None,
        prebuilt_cohort: Optional[Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]] = None,
    ) -> None:


        """Initializes the HealthDataModule.

        Args:
            aggregator (LabelAggregator): Strategy for aggregating multiple survey answers into one label.
            sampler (TimeSampler): Strategy for sampling time-series data relative to survey timestamps.
            scaler (Optional[Any]): Optional scaler object (e.g. sklearn StandardScaler) to apply to features.
            preprocessors (Optional[Dict[str, Any]]): Dictionary of preprocessor objects for data modalities.
            modalities (List[str]): List of database tables to fetch (e.g. ["step", "calorie"]).
            batch_size (int): Number of samples per batch.

            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
            train_val_test_split (Tuple[float, float, float]): Fraction of data for train, validation, and test sets.
            random_state (int): Seed for reproducibility.
            split_mode (Literal["user", "longitudinal"]): The strategy for splitting the data.
                - "user": Split by user ID (ensures disjoint populations). Matches
                  deployment, where a new user of the app has no labels at all.
                - "longitudinal": Temporal split per user (predict future from past).
                Row-level random splitting was removed; see CohortSplitter for why.
            collapse_strategy (str): Aggregation strategy to collapse multiple daily survey responses for non-yes-no questions.
            use_demographics (bool): Whether to query, process, and pass user demographics as context.
            enrollment_lead_days (float): Days of sensor data retained before each user's
                first survey. Must exceed the sampler's lookback so early surveys still
                see a full window.
            enrollment_trail_days (float): Days of sensor data retained after each user's
                last survey.
            require_sensor_data (bool): Drop samples whose sampling window contains no
                sensor records at all. Keeping them trains the model on all-zero
                sequences that are indistinguishable from genuine inactivity, and
                holding the cohort fixed across feature sets is what makes a
                sensors-vs-demographics comparison meaningful.
            use_survey_context (bool): Append per-response context (which daily survey
                this is, and how long its "since last prompt" referent ran) to the
                demographics vector.
            include_time_features (Optional[bool]): Optional override for inclusion of
                cyclical time features in the sequence sampler.
            exclude_user_ids (Optional[List[int]]): Users dropped from every stream.
                Defaults to CohortBuilder's test/discontinued accounts.
        """
        super().__init__()
        if include_time_features is not None and hasattr(sampler, "include_time_features"):
            sampler.include_time_features = include_time_features

        self.save_hyperparameters(logger=False)
        self.prebuilt_cohort = prebuilt_cohort

        # Build the modality_cols mapping for the requested modalities
        self.hparams.modality_cols = {
            mod: self._MODALITY_MAPPING.get(mod, mod) 
            for mod in self.hparams.modalities
        }

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # List of question IDs that are asked twice daily (present in both morning and afternoon surveys)
        self.ask_twice_question_ids = [
            2, 3, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18,
            21, 22, 23, 24, 25, 26, 27, 28,
            31, 32, 33, 34, 35, 36, 37,
            47, 48, 49, 50, 51, 52
        ]

    def get_prebuilt_cohort(self) -> Optional[Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]]:
        """Returns shallow copies of the extracted, coverage-filtered cohort dataframes."""
        return getattr(self, "raw_cohort", None)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data for all stages.

        This method performs the following steps:
            1. Orchestrates cohort extraction via CohortBuilder (or uses prebuilt_cohort).
            2. Splits the cohort into training, validation, and test datasets.
            3. Fits the feature scaler on the training data.
            4. Instantiates the final HealthDataset objects.

        Args:
            stage: The stage for which to setup data (fit, validate, test, predict).
        """
        if not self.data_train and not self.data_val and not self.data_test:
            if self.prebuilt_cohort is not None:
                m_dfs_raw, master_raw, demo_raw = self.prebuilt_cohort
                modality_dfs = {k: df.copy() for k, df in m_dfs_raw.items()}
                master_df = master_raw.copy()
                demographics_df = demo_raw.copy()
            else:
                builder = CohortBuilder(
                    modalities=self.hparams.modalities,
                    modality_cols=self.hparams.modality_cols,
                    preprocessors=self.hparams.preprocessors,
                    aggregator=self.hparams.aggregator,
                    os_filter=self.hparams.os_filter,
                    collapse_strategy=self.hparams.collapse_strategy,
                    use_demographics=self.hparams.use_demographics,
                    use_sleep=self.hparams.use_sleep,
                    enrollment_lead_days=self.hparams.enrollment_lead_days,
                    enrollment_trail_days=self.hparams.enrollment_trail_days,
                    exclude_user_ids=self.hparams.exclude_user_ids,
                )
                modality_dfs, master_df, demographics_df = builder.build()

                # Drop samples with no sensor coverage before splitting, so that every
                # split reflects the cohort the model is actually trained and scored on
                if self.hparams.require_sensor_data:
                    master_df = self._filter_to_covered_samples(master_df, modality_dfs)

                self.raw_cohort = (
                    {k: df.copy() for k, df in modality_dfs.items()},
                    master_df.copy(),
                    demographics_df.copy(),
                )

            self.master_df = master_df

            # Perform splitting based on the selected evaluation strategy
            train_df, val_df, test_df = self._split_data(master_df)

            # Standardize per-response survey context on training statistics only
            if self.hparams.use_survey_context:
                train_df, val_df, test_df = self._add_survey_context(train_df, val_df, test_df)

            # Fit sleep feature scaling and apply standardizations
            if self.hparams.use_sleep:
                sleep_train = pd.to_numeric(train_df["sleep_hours"], errors="coerce").dropna()
                self.sleep_mean = sleep_train.mean() if not sleep_train.empty else 7.0
                self.sleep_std = sleep_train.std() if not sleep_train.empty and sleep_train.std() > 0 else 2.0

                for df in [train_df, val_df, test_df]:
                    df["sleep_hours_scaled"] = (pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(self.sleep_mean) - self.sleep_mean) / self.sleep_std
                    df["sleep_class_0"] = (df["sleep_category"] == 0).astype(float)
                    df["sleep_class_1"] = (df["sleep_category"] == 1).astype(float)
                    df["sleep_class_2"] = (df["sleep_category"] == 2).astype(float)
                    df["sleep_unknown"] = df["sleep_category"].isna().astype(float)

            # Process demographics and device source embeddings
            demo_processor = DemographicsProcessor(use_demographics=self.hparams.use_demographics)
            self.demographics_map, self.default_demographics = demo_processor.fit_transform(
                train_df=train_df,
                demographics_df=demographics_df,
                modality_dfs=modality_dfs,
                master_df=master_df,
            )
            self.demographics_dim = demo_processor.demographics_dim
            if self.hparams.use_sleep:
                self.demographics_dim += 5
            if self.hparams.use_survey_context:
                self.demographics_dim += 3

            # If the sampler needs access to the labels (e.g. LagSampler), provide them now
            if hasattr(self.hparams.sampler, "set_labels"):
                self.hparams.sampler.set_labels(master_df)

            # Create user_to_idx mapping based on training users
            # Reserve 0 for unseen/average users
            train_users = train_df["app_user_id"].unique()
            self.user_to_idx = {uid: idx + 1 for idx, uid in enumerate(train_users)}

            # Fit the scaler on training sequences only
            if self.hparams.scaler is not None and hasattr(self.hparams.scaler, "fit"):
                self._fit_scaler(train_df, modality_dfs)

            # Instantiate Dataset objects
            is_regression = getattr(self.hparams.aggregator, "is_regression", False)
            self.data_train = HealthDataset(
                train_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                demographics_map=self.demographics_map,
                default_demographics=self.default_demographics,
                use_sleep=self.hparams.use_sleep,
                use_survey_context=self.hparams.use_survey_context,
            )
            self.data_val = HealthDataset(
                val_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                demographics_map=self.demographics_map,
                default_demographics=self.default_demographics,
                use_sleep=self.hparams.use_sleep,
                use_survey_context=self.hparams.use_survey_context,
            )
            self.data_test = HealthDataset(
                test_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                demographics_map=self.demographics_map,
                default_demographics=self.default_demographics,
                use_sleep=self.hparams.use_sleep,
                use_survey_context=self.hparams.use_survey_context,
            )

    def _filter_to_covered_samples(
        self, master_df: pd.DataFrame, modality_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Drops responses whose sampling window contains no sensor records.

        Sensor missingness here is block-structured by user rather than scattered:
        a person either syncs health data or goes days without it. A sample with
        an empty window carries no sensor information, so keeping it trains the
        model on an all-zero sequence that looks identical to genuine inactivity.

        Samplers that read no sensor data (e.g. ``LagSampler``) report no window
        and are left untouched.
        """
        sampler = self.hparams.sampler
        if not hasattr(sampler, "window_bounds") or master_df.empty:
            return master_df

        probe = sampler.window_bounds(master_df.iloc[0]["record_timestamp"])
        if probe is None:
            return master_df

        # Precompute window bounds for all survey responses
        timestamps = master_df["record_timestamp"].values
        bounds_list = [sampler.window_bounds(ts) for ts in timestamps]

        w_starts = np.array(
            [np.datetime64(pd.Timestamp(b[0])) if b is not None else np.datetime64("NaT") for b in bounds_list]
        )
        w_ends = np.array(
            [np.datetime64(pd.Timestamp(b[1])) if b is not None else np.datetime64("NaT") for b in bounds_list]
        )

        valid_mask = ~np.isnat(w_starts)
        keep = ~valid_mask  # Retain responses where sampler window_bounds returned None

        # Group modality interval arrays by user for 2D vectorized broadcasting
        user_intervals: Dict[Any, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for df in modality_dfs.values():
            if df.empty or "app_user_id" not in df.columns:
                continue
            starts_col = df["start_timestamp"]
            ends_col = df["end_timestamp"] if "end_timestamp" in df.columns else starts_col
            for uid, idx in df.groupby("app_user_id").groups.items():
                s = starts_col.loc[idx].values.astype("datetime64[ns]")
                e = ends_col.loc[idx].values.astype("datetime64[ns]")
                user_intervals.setdefault(uid, []).append((s, e))

        user_ids = master_df["app_user_id"].values
        for uid, intervals in user_intervals.items():
            user_indices = np.where((user_ids == uid) & valid_mask)[0]
            if len(user_indices) == 0:
                continue

            u_starts = w_starts[user_indices, None]  # Shape: [N_responses, 1]
            u_ends = w_ends[user_indices, None]      # Shape: [N_responses, 1]

            u_keep = np.zeros(len(user_indices), dtype=bool)
            for s, e in intervals:
                # Broadcasting: (sensor_ends > window_starts) & (sensor_starts < window_ends)
                # s, e shape: [1, M_sensor_records]
                s_mat = s[None, :]
                e_mat = e[None, :]
                overlap = (e_mat > u_starts) & (s_mat < u_ends)
                u_keep |= overlap.any(axis=1)

            keep[user_indices] = u_keep

        n_dropped = int((~keep).sum())
        if n_dropped:
            import logging
            logging.getLogger(__name__).warning(
                f"Dropping {n_dropped} of {len(master_df)} responses with no sensor "
                f"coverage in the sampling window ({n_dropped / len(master_df):.1%})."
            )
        return master_df.loc[keep].reset_index(drop=True)

    def _add_survey_context(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Builds per-response context features, scaled on training data only.

        The two daily surveys describe different periods — a morning response
        covers overnight, an afternoon response covers the day so far — and the
        gap back to the previous response varies widely when surveys are missed.
        Both are properties of the sample rather than the person, so they are
        carried alongside demographics instead of being averaged away.

        Adds ``is_morning``, ``referent_hours_scaled`` and ``referent_missing``
        (set for a user's first response, which has no preceding prompt).
        """
        ref_train = pd.to_numeric(train_df.get("referent_hours"), errors="coerce").dropna()
        self.referent_mean = float(ref_train.mean()) if not ref_train.empty else 12.0
        std = float(ref_train.std()) if not ref_train.empty else 0.0
        self.referent_std = std if std > 0 else 6.0

        out = []
        for df in (train_df, val_df, test_df):
            df = df.copy()
            ref = pd.to_numeric(df.get("referent_hours"), errors="coerce")
            df["referent_missing"] = ref.isna().astype(float)
            df["referent_hours_scaled"] = (
                ref.fillna(self.referent_mean) - self.referent_mean
            ) / self.referent_std
            if "is_morning" not in df.columns:
                df["is_morning"] = 0.0
            df["is_morning"] = df["is_morning"].astype(float)
            out.append(df)
        return out[0], out[1], out[2]

    def _fit_scaler(self, train_df: pd.DataFrame, modality_dfs: Dict[str, pd.DataFrame]) -> None:
        """Fits the configured scaler using data sampled from the training dataframe."""
        modalities = sorted(list(modality_dfs.keys()))
        modality_cols = self.hparams.modality_cols
        train_seqs = [
            self.hparams.sampler(
                survey_timestamp=row["record_timestamp"],
                app_user_id=row["app_user_id"],
                modality_dfs=modality_dfs,
                modality_cols=modality_cols,
                modalities=modalities,
            )
            for _, row in train_df.iterrows()
        ]
        if train_seqs:
            if hasattr(self.hparams.scaler, "fit_by_subject"):
                user_ids = train_df["app_user_id"].values
                self.hparams.scaler.fit_by_subject(train_seqs, user_ids)
            else:
                stacked = np.concatenate(train_seqs, axis=0)  # [N*Time, Features]
                self.hparams.scaler.fit(stacked)

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Dispatches to the appropriate splitting strategy.

        Args:
            df: The master linked dataframe to split.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        splitter = CohortSplitter(
            split_mode=self.hparams.split_mode,
            train_val_test_split=self.hparams.train_val_test_split,
            random_state=self.hparams.random_state,
        )
        return splitter.split(df)

    def train_dataloader(self) -> DataLoader:
        """Returns the training data loader with shuffling enabled."""
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=True,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation data loader."""
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test data loader."""
        return DataLoader(
            dataset=self.data_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False
        )
