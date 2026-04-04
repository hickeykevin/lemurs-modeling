import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List

class HealthDataset(Dataset):
    """
    General Multimodal Dataset for health metrics.
    Takes pre-linked survey-to-user data and a dictionary of modality dataframes.
    
    Produces tensors of shape [Time=24, Features=N_modalities]
    """
    def __init__(
        self, 
        linked_data: pd.DataFrame, 
        modality_dfs: Dict[str, pd.DataFrame],
        modality_cols: Dict[str, str]
    ):
        """
        :param linked_data: DataFrame of survey records.
        :param modality_dfs: Dictionary mapping modality name (e.g. 'step') to its DataFrame.
        :param modality_cols: Dictionary mapping modality name to its numeric column (e.g. {'step': 'steps'}).
        """
        self.data_links = linked_data.reset_index(drop=True)
        self.modality_dfs = modality_dfs
        self.modality_cols = modality_cols
        self.modalities = sorted(list(modality_dfs.keys())) # Ensure consistent order

    def __len__(self):
        return len(self.data_links)

    def __getitem__(self, idx):
        row = self.data_links.iloc[idx]
        app_user_id = row['app_user_id']
        survey_timestamp = row['record_timestamp']
        lookback_date = survey_timestamp.date() - timedelta(days=1)
        
        full_range = pd.date_range(
            start=pd.Timestamp(lookback_date),
            periods=24,
            freq='h'
        )

        all_modality_features = []

        for mod in self.modalities:
            df = self.modality_dfs[mod]
            val_col = self.modality_cols[mod]
            
            # 1. Filter for user/day
            # Note: We assume all health tables have 'app_user_id' and 'start_timestamp'
            mask = (df['app_user_id'] == app_user_id) & (df['start_timestamp'].dt.date == lookback_date)
            day_data = df[mask].copy()

            if day_data.empty:
                resampled = np.zeros(24, dtype=np.float32)
            else:
                # 2. Hourly Resampling
                day_data['hour'] = day_data['start_timestamp'].dt.floor('h')
                resampled = (day_data.groupby('hour')[val_col]
                             .sum()
                             .reindex(full_range, fill_value=0)
                             .values.astype(np.float32))
            
            all_modality_features.append(resampled)

        # Combine modalities into [Time=24, Features=N]
        # shape: (24, num_modalities)
        sequence = np.stack(all_modality_features, axis=-1)
            
        target = row['answer']
        return torch.tensor(sequence), torch.tensor(target, dtype=torch.long)
