import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List
from src.data.components.samplers import TimeSampler

class HealthDataset(Dataset):
    """
    General Multimodal Dataset for health metrics.
    Takes pre-linked survey-to-user data, a dictionary of modality dataframes,
    and a sampler strategy to generate feature sequences.
    """
    def __init__(
        self, 
        linked_data: pd.DataFrame, 
        modality_dfs: Dict[str, pd.DataFrame],
        modality_cols: Dict[str, str],
        sampler: TimeSampler
    ):
        """
        :param linked_data: DataFrame of survey records.
        :param modality_dfs: Dictionary mapping modality name (e.g. 'step') to its DataFrame.
        :param modality_cols: Dictionary mapping modality name to its numeric column (e.g. {'step': 'steps'}).
        :param sampler: Strategy for sampling time-series data relative to survey timestamps.
        """
        self.data_links = linked_data.reset_index(drop=True)
        self.modality_dfs = modality_dfs
        self.modality_cols = modality_cols
        self.sampler = sampler
        self.modalities = sorted(list(modality_dfs.keys())) # Ensure consistent order

    def __len__(self):
        return len(self.data_links)

    def __getitem__(self, idx):
        row = self.data_links.iloc[idx]
        app_user_id = row['app_user_id']
        survey_timestamp = row['record_timestamp']
        
        # Use the sampler strategy to get the feature matrix
        sequence = self.sampler(
            survey_timestamp=survey_timestamp,
            app_user_id=app_user_id,
            modality_dfs=self.modality_dfs,
            modality_cols=self.modality_cols,
            modalities=self.modalities
        )
            
        target = row['answer']
        return torch.tensor(sequence), torch.tensor(target, dtype=torch.long)
