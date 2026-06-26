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
        split_mode: Literal["random", "user", "longitudinal"] = "random",
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "mean",
        use_prev_prediction: bool = False,
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
            split_mode (Literal["random", "user", "longitudinal"]): The strategy for splitting the data.
                - "random": Standard row-level random shuffle.
                - "user": Split by user ID (ensures disjoint populations).
                - "longitudinal": Temporal split per user (predict future from past).
            collapse_strategy (str): Aggregation strategy to collapse multiple daily survey responses for non-yes-no questions.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data for all stages.

        This method performs the following steps:
            1. Orchestrates cohort extraction via CohortBuilder.
            2. Splits the cohort into training, validation, and test datasets.
            3. Fits the feature scaler on the training data.
            4. Instantiates the final HealthDataset objects.

        Args:
            stage: The stage for which to setup data (fit, validate, test, predict).
        """
        if not self.data_train and not self.data_val and not self.data_test:
            builder = CohortBuilder(
                modalities=self.hparams.modalities,
                modality_cols=self.hparams.modality_cols,
                preprocessors=self.hparams.preprocessors,
                aggregator=self.hparams.aggregator,
                os_filter=self.hparams.os_filter,
                collapse_strategy=self.hparams.collapse_strategy
            )
            modality_dfs, master_df = builder.build()
            
            self.master_df = master_df

            # Perform splitting based on the selected evaluation strategy
            train_df, val_df, test_df = self._split_data(master_df)

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

            # Sort chronologically and precompute link mappings if use_prev_prediction is True
            if self.hparams.use_prev_prediction:
                train_df = train_df.sort_values(['app_user_id', 'record_timestamp']).reset_index(drop=True)
                val_df = val_df.sort_values(['app_user_id', 'record_timestamp']).reset_index(drop=True)
                test_df = test_df.sort_values(['app_user_id', 'record_timestamp']).reset_index(drop=True)
                
                for df in [train_df, val_df, test_df]:
                    prev_indices = []
                    for idx in range(len(df)):
                        if idx > 0 and df.iloc[idx - 1]['app_user_id'] == df.iloc[idx]['app_user_id']:
                            prev_indices.append(idx - 1)
                        else:
                            prev_indices.append(-1)
                    df['prev_sample_idx'] = prev_indices

            # Instantiate Dataset objects
            is_regression = getattr(self.hparams.aggregator, "is_regression", False)
            self.data_train = HealthDataset(
                train_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                use_prev_prediction=self.hparams.use_prev_prediction
            )
            self.data_val = HealthDataset(
                val_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                use_prev_prediction=self.hparams.use_prev_prediction
            )
            self.data_test = HealthDataset(
                test_df, modality_dfs, self.hparams.modality_cols,
                self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                is_regression=is_regression,
                use_prev_prediction=self.hparams.use_prev_prediction
            )

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
        split_strategies = {
            "random": self._split_random,
            "user": self._split_by_user,
            "longitudinal": self._split_longitudinally,
        }
        
        strategy_fn = split_strategies.get(self.hparams.split_mode)
        if not strategy_fn:
            raise ValueError(
                f"Unknown split_mode: {self.hparams.split_mode}. "
                f"Available modes: {list(split_strategies.keys())}"
            )
            
        return strategy_fn(df)

    def _split_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a standard row-level random shuffle split."""
        train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio), 
            random_state=self.hparams.random_state
        )
        
        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=val_ratio_relative, 
            random_state=self.hparams.random_state
        )
        return train_df, val_df, test_df

    def _split_by_user(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a split at the user level to ensure disjoint populations."""
        train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
        unique_users = df['app_user_id'].unique()
        
        train_users, temp_users = train_test_split(
            unique_users, 
            test_size=(1 - train_ratio), 
            random_state=self.hparams.random_state
        )
        
        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_users, test_users = train_test_split(
            temp_users, 
            train_size=val_ratio_relative, 
            random_state=self.hparams.random_state
        )
        
        train_df = df[df['app_user_id'].isin(train_users)]
        val_df = df[df['app_user_id'].isin(val_users)]
        test_df = df[df['app_user_id'].isin(test_users)]
        return train_df, val_df, test_df

    def _split_longitudinally(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a temporal split within each individual user's history."""
        train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
        train_list, val_list, test_list = [], [], []
        
        for _, group in df.groupby('app_user_id'):
            group = group.sort_values('record_timestamp')
            n = len(group)
            
            if n < 3: 
                # Not enough data to split 3 ways, default to training
                train_list.append(group)
                continue
                
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # Ensure at least one sample in each set where possible
            train_end = max(1, train_end)
            val_end = max(train_end + 1, val_end)
            
            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])
        
        return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

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
