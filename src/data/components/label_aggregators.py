import pandas as pd
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Union

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

    @property
    def is_regression(self) -> bool:
        """Indicates whether this aggregator outputs a continuous regression target."""
        return False

class MeanAggregator(LabelAggregator):
    """Aggregates answers by taking the mean and optionally binarizing."""
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None):
        self.question_ids = question_ids if question_ids is not None else [2, 3]
        self.threshold = threshold

    def get_question_ids(self) -> list:
        return self.question_ids
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].mean().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg

class MaxAggregator(LabelAggregator):
    """Aggregates answers by taking the max and optionally binarizing."""
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None):
        self.question_ids = question_ids if question_ids is not None else [2, 3]
        self.threshold = threshold

    def get_question_ids(self) -> list:
        return self.question_ids
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('survey_response_id')['answer'].max().reset_index()
        if self.threshold is not None:
            agg['answer'] = (agg['answer'] >= self.threshold).astype(int)
        return agg



class RuleBasedAggregator(LabelAggregator):
    """Aggregates survey answers based on configurable clinical rules.

    This aggregator evaluates subsets of questions according to specific operators
    (e.g., 'sum', 'ge', 'any_eq') and combines the results using 'any' (OR) or 
    'all' (AND) logic to yield binary risk labels.
    """
    
    def __init__(self, rules: Union[list, dict, Mapping], combination_logic: str = "any") -> None:
        """Initializes the RuleBasedAggregator.

        Args:
            rules (list | dict | Mapping): A list or dictionary of dictionaries defining evaluation clauses. 
                Each rule should contain:
                - 'ids' (list[int]): Question IDs to target.
                - 'op' (str): Operator to apply. Options include:
                  'sum', 'mean', 'ge', 'any_eq', 'sum_le'.
                - 'threshold' or 'val' (float/int): Value to compare against.
            combination_logic (str, optional): Strategy for merging multiple rules.
                'any' evaluates to True if at least one rule passes.
                'all' requires every rule to pass. Defaults to "any".
        """
        if isinstance(rules, (dict, Mapping)):
            self.rules = list(rules.values())
        else:
            self.rules = rules
        self.combination_logic = combination_logic

    def get_question_ids(self) -> list:
        """Retrieves all unique question IDs targeted across active rules.

        Returns:
            list[int]: Unique targeted question IDs.
        """
        ids = set()
        for rule in self.rules:
            ids.update(rule['ids'])
        return list(ids)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates rule logic against patient survey responses.

        Args:
            df (pd.DataFrame): Long-format answers with 'survey_response_id', 
                'question_id', and 'answer'.

        Returns:
            pd.DataFrame: Aggregated binary status per 'survey_response_id'.
        """

        if df.empty:
            return pd.DataFrame(columns=['survey_response_id', 'answer'])
            
        pivoted = df.pivot(index='survey_response_id', columns='question_id', values='answer')
        
        results = []
        for rule in self.rules:
            ids = [c for c in rule['ids'] if c in pivoted.columns]
            if not ids:
                continue
                
            op = rule['op']
            val = rule.get('val', rule.get('threshold'))
            if op == "mean":
                score = pivoted[ids].mean(axis=1)
                condition = score >= val
            elif op == "sum":
                score = pivoted[ids].sum(axis=1)
                condition = score >= val
            elif op == "sum_le":
                score = pivoted[ids].sum(axis=1)
                condition = score <= val
            elif op == "max":
                score = pivoted[ids].max(axis=1)
                condition = score >= val
            elif op == "ge":
                condition = (pivoted[ids] >= val).any(axis=1)
            elif op == "any_eq":
                condition = (pivoted[ids] == val).any(axis=1)
            else:
                raise ValueError(f"Unknown operation: {op}")

                
            results.append(condition)
            
        if not results:
            return pd.DataFrame({'survey_response_id': df['survey_response_id'].unique(), 'answer': 0})
            
        combined = results[0]
        for cond in results[1:]:
            if self.combination_logic == 'any':
                combined = combined | cond
            elif self.combination_logic == 'all':
                combined = combined & cond
                
        return pd.DataFrame({
            'survey_response_id': pivoted.index,
            'answer': combined.astype(int)
        }).reset_index(drop=True)


class RegressionAggregator(LabelAggregator):
    """Aggregates survey answers by summing Likert scale questions to produce a continuous target."""
    
    def __init__(self, likert_ids: list, shift_likert: bool = True) -> None:
        """Initializes the RegressionAggregator.

        Args:
            likert_ids (list[int]): Question IDs to target and sum.
            shift_likert (bool): If True, shifts 1-indexed Likert scales to start at 0 by subtracting 1.
        """
        self.likert_ids = likert_ids
        self.shift_likert = shift_likert

    def get_question_ids(self) -> list:
        """Retrieves targeted question IDs.

        Returns:
            list[int]: Targeted question IDs.
        """
        return list(set(self.likert_ids))

    @property
    def is_regression(self) -> bool:
        return True

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sums target question answers per response.

        Args:
            df (pd.DataFrame): Long-format answers with 'survey_response_id', 
                'question_id', and 'answer'.

        Returns:
            pd.DataFrame: Aggregated continuous status per 'survey_response_id'.
        """
        if df.empty:
            return pd.DataFrame(columns=['survey_response_id', 'answer'])
            
        pivoted = df.pivot(index='survey_response_id', columns='question_id', values='answer')
        
        likert_cols = [c for c in self.likert_ids if c in pivoted.columns]
        if likert_cols:
            likert_scores = pivoted[likert_cols]
            if self.shift_likert:
                # Check if values are already 0-indexed (i.e. min value is 0)
                # We skip shifting if the min value is 0 to avoid double-shifting.
                min_val = likert_scores.min(skipna=True).min(skipna=True)
                if min_val == 0:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Dataset answers already contain 0. Skipping Likert shifting to avoid double-shifting."
                    )
                else:
                    likert_scores = (likert_scores - 1).clip(lower=0)
            # Use sum with min_count=1 to ensure that if all answers are NaN, we get NaN instead of 0
            total_score = likert_scores.sum(axis=1, min_count=1)
        else:
            total_score = pd.Series(np.nan, index=pivoted.index)
            
        return pd.DataFrame({
            'survey_response_id': total_score.index,
            'answer': total_score.values
        }).reset_index(drop=True)



