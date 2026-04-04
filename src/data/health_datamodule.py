from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.database_service import DatabaseService
from src.data.components.health_dataset import HealthDataset
import pandas as pd
from sklearn.model_selection import train_test_split

class HealthDataModule(LightningDataModule):
    """
    `LightningDataModule` for Multimodal Health data prediction.
    """
    def __init__(
        self,
        modalities: List[str] = ["step"],
        modality_cols: Dict[str, str] = {"step": "steps"},
        question_id: int = 2,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
    ) -> None:
        """
        :param modalities: List of database tables to fetch (e.g. ["step", "calorie"]).
        :param modality_cols: Mapping of table names to numeric value column (e.g. {'step':'steps', 'calorie':'calories'}).
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            db = DatabaseService()
            if not db.connect():
                raise Exception("Failed to connect to the database.")

            try:
                # 1. Fetch data from DB for all modalities
                modality_dfs = {}
                for mod in self.hparams.modalities:
                    df = db.extract_from_database(mod)
                    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                    modality_dfs[mod] = df

                # 2. Extract survey data
                answer_df = db.extract_from_database("answer")
                survey_response_df = db.extract_from_database("survey_response")
                survey_response_df['timestamp'] = pd.to_datetime(survey_response_df['timestamp'])
            finally:
                db.disconnect()

            # 3. Global Pre-processing and Linking (based on survey responses)
            target_answers = answer_df[answer_df['question_id'] == self.hparams.question_id].copy()
            master_df = pd.merge(
                target_answers,
                survey_response_df[['id', 'app_user_id', 'timestamp']],
                left_on='survey_response_id',
                right_on='id',
                suffixes=('', '_survey')
            ).rename(columns={'timestamp': 'record_timestamp'})

            master_df['answer'] = pd.to_numeric(master_df['answer'], errors='coerce')
            master_df = master_df.dropna(subset=['answer'])
            master_df['answer'] = master_df['answer'].astype(int)

            # 4. Split and instantiate datasets
            train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
            train_df, temp_df = train_test_split(
                master_df, test_size=(1 - train_ratio), random_state=self.hparams.random_state
            )
            
            val_ratio_relative = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df, train_size=val_ratio_relative, random_state=self.hparams.random_state
            )

            # Modality specific columns needed for the dataset
            self.data_train = HealthDataset(train_df, modality_dfs, self.hparams.modality_cols)
            self.data_val = HealthDataset(val_df, modality_dfs, self.hparams.modality_cols)
            self.data_test = HealthDataset(test_df, modality_dfs, self.hparams.modality_cols)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False)
