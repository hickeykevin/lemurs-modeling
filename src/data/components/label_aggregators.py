import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

class LabelAggregator(ABC):
    """Base class for survey answer aggregation strategies."""
    
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe of raw answers into a dataframe of aggregated labels.
        
        Args:
            df: Dataframe with 'survey_response_id' and 'answer' columns.
            
        Returns:
            Dataframe with 'survey_response_id' and the final 'answer' column.
        """
        pass

class MeanAggregator(LabelAggregator):
    """Aggregates answers by taking the mean and optionally binarizing."""
    
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].mean().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg

class MaxAggregator(LabelAggregator):
    """Aggregates answers by taking the max and optionally binarizing."""
    
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].max().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg
