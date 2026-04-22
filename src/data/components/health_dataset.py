import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from src.data.components.samplers import TimeSampler

class HealthDataset(Dataset):
    """General Multimodal Dataset for health metrics.
    
    This dataset takes pre-linked survey-to-user mappings and a dictionary of raw modality 
    dataframes. It utilizes a `TimeSampler` strategy to slice and resample the time-series 
    data into fixed-length sequences suitable for model training.

    Attributes:
        data_links (pd.DataFrame): DataFrame containing the link between survey responses and users.
        modality_dfs (Dict[str, pd.DataFrame]): Dictionary mapping modality names to their dataframes.
        modality_cols (Dict[str, str]): Mapping of modality names to their numeric column names.
        sampler (TimeSampler): Strategy for generating time-series sequences relative to surveys.
        scaler (Optional[Any]): Optional fitted scaler to normalize features.
        modalities (List[str]): Sorted list of modality names to ensure consistent feature ordering.
    """

    def __init__(
        self, 
        linked_data: pd.DataFrame, 
        modality_dfs: Dict[str, pd.DataFrame],
        modality_cols: Dict[str, str],
        sampler: TimeSampler,
        scaler: Optional[Any] = None
    ):
        """Initializes the HealthDataset.

        Args:
            linked_data (pd.DataFrame): DataFrame of survey records with app_user_id and record_timestamp.
            modality_dfs (Dict[str, pd.DataFrame]): Dictionary of health dataframes (e.g. {'step': df}).
            modality_cols (Dict[str, str]): Column names for values (e.g. {'step': 'steps'}).
            sampler (TimeSampler): The sampling strategy to use (Rolling, Offset, etc).
            scaler (Optional[Any]): A pre-fitted scaler object (e.g. StandardScaler).
        """
        self.data_links = linked_data.reset_index(drop=True)
        self.modality_dfs = modality_dfs
        self.modality_cols = modality_cols
        self.sampler = sampler
        self.scaler = scaler
        self.modalities = sorted(list(modality_dfs.keys())) # Ensure consistent feature order

    def __len__(self) -> int:
        """Returns the total number of survey records in the dataset."""
        return len(self.data_links)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates one sample from the dataset.

        Args:
            idx (int): The index of the record to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - features: A tensor of shape [Time, Modalities]
                - target: The aggregated survey answer as a long tensor.
        """
        row = self.data_links.iloc[idx]
        app_user_id = row['app_user_id']
        survey_timestamp = row['record_timestamp']
        
        # 1. Slice and resample time-series data using the sampler strategy
        # Resulting sequence shape: [Time, Modalities]
        sequence = self.sampler(
            survey_timestamp=survey_timestamp,
            app_user_id=app_user_id,
            modality_dfs=self.modality_dfs,
            modality_cols=self.modality_cols,
            modalities=self.modalities
        )
        
        # 2. Apply feature scaling if a scaler was provided
        if self.scaler is not None:
            # Reshape to [Time * 1, Modalities] as sklearn scalers expect [Samples, Features]
            original_shape = sequence.shape
            sequence = self.scaler.transform(sequence.reshape(-1, original_shape[-1]))
            sequence = sequence.reshape(original_shape)
            
        target = row['answer']
        
        # 3. Convert to torch tensors
        return (
            torch.tensor(sequence, dtype=torch.float32), 
            torch.tensor(target, dtype=torch.long)
        )
