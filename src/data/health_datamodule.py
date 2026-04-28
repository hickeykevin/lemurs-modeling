from typing import Any, Dict, Optional, Tuple, List, Literal
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.database_service import DatabaseService
from src.data.components.health_dataset import HealthDataset
from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
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
        modalities: List[str] = ["step"],
        batch_size: int = 8,

        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        split_mode: Literal["random", "user", "longitudinal"] = "random",
    ) -> None:

        """Initializes the HealthDataModule.

        Args:
            aggregator (LabelAggregator): Strategy for aggregating multiple survey answers into one label.
            sampler (TimeSampler): Strategy for sampling time-series data relative to survey timestamps.
            scaler (Optional[Any]): Optional scaler object (e.g. sklearn StandardScaler) to apply to features.
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data for all stages.

        This method performs the following steps:
            1. Connects to the database and extracts health metrics and survey data.
            2. Aggregates target labels using the configured `aggregator` strategy.
            3. Links survey answers to their corresponding users and timestamps.
            4. Performs data splitting based on the configured `split_mode`.
            5. Instantiates the final `HealthDataset` objects.

        Args:
            stage: The stage for which to setup data (fit, validate, test, predict).
        """
        if not self.data_train and not self.data_val and not self.data_test:
            db = DatabaseService()
            if not db.connect():
                raise Exception("Failed to connect to the database.")

            try:
                # 1. Fetch raw data for all requested modalities
                modality_dfs = {}
                for mod in self.hparams.modalities:
                    df = db.extract_from_database(mod)
                    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                    modality_dfs[mod] = df

                # 2. Extract survey response labels
                answer_df = db.extract_from_database("answer")
                survey_response_df = db.extract_from_database("survey_response")
                survey_response_df['timestamp'] = pd.to_datetime(survey_response_df['timestamp'])
            finally:
                db.disconnect()

            # 3. Aggregate and link survey answers
            # Extract question IDs required by the chosen aggregator
            question_ids = self.hparams.aggregator.get_question_ids()

            # Filter for requested questions and ensure numeric values
            target_answers = answer_df[answer_df['question_id'].isin(question_ids)].copy()


            target_answers['answer'] = pd.to_numeric(target_answers['answer'], errors='coerce')
            target_answers = target_answers.dropna(subset=['answer'])

            # Delegate aggregation logic to the strategy object
            aggregated_answers = self.hparams.aggregator(target_answers)

            # Join with survey_response metadata to get user IDs and timestamps
            master_df = pd.merge(
                aggregated_answers,
                survey_response_df[['id', 'app_user_id', 'timestamp']],
                left_on='survey_response_id',
                right_on='id',
                suffixes=('', '_survey')
            ).rename(columns={'timestamp': 'record_timestamp'})
            # 4. Perform splitting based on the selected evaluation strategy
            train_df, val_df, test_df = self._split_data(master_df)

            # 5. Optional Normalization
            # We fit the scaler on the training data ONLY
            if self.hparams.scaler is not None and hasattr(self.hparams.scaler, "fit"):
                # Create a temporary dataset to collect training features
                temp_train_ds = HealthDataset(
                    train_df, modality_dfs, self.hparams.modality_cols, self.hparams.sampler
                )
                all_features = []
                for i in range(len(temp_train_ds)):
                    seq, _ = temp_train_ds[i]
                    all_features.append(seq.numpy())
                
                all_features_flattened = np.concatenate(all_features, axis=0) # [N*Time, Features]
                self.hparams.scaler.fit(all_features_flattened)

            # 6. Instantiate final Dataset objects with the appropriately filtered data
            self.data_train = HealthDataset(
                train_df, modality_dfs, self.hparams.modality_cols, 
                self.hparams.sampler, self.hparams.scaler
            )
            self.data_val = HealthDataset(
                val_df, modality_dfs, self.hparams.modality_cols, 
                self.hparams.sampler, self.hparams.scaler
            )
            self.data_test = HealthDataset(
                test_df, modality_dfs, self.hparams.modality_cols, 
                self.hparams.sampler, self.hparams.scaler
            )

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
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation data loader."""
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False
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
