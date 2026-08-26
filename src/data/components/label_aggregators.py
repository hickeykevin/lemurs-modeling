import pandas as pd
import numpy as np
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

    @property
    def num_classes(self) -> int:
        """Returns the number of target classes for classification, or 1 for regression."""
        return 1 if self.is_regression else 2

class MeanAggregator(LabelAggregator):
    """Aggregates answers by taking the mean and optionally binarizing."""
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None, **kwargs):
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
    
    def __init__(self, question_ids: Optional[list] = None, threshold: Optional[float] = None, **kwargs):
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
    
    def __init__(self, rules: Union[list, dict, Mapping], combination_logic: str = "any", **kwargs) -> None:
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

            # The DB can return 'answer' as a string column (observed for
            # both Likert-scale questions returning '1'..'5' as text, and
            # yes/no questions returning literal "yes"/"no") even though
            # every op below except "any_eq" compares against a numeric
            # val. An unconverted string-vs-float compare doesn't raise --
            # pandas' string dtypes silently produce a comparison result
            # that is not the intended one, which can flip a rule to "true
            # for nearly every response" without any visible error.
            #
            # Map yes/no text to 1/0 first (case-insensitive; other values
            # pass through untouched), THEN coerce to numeric whenever val
            # itself is numeric (every op here except a string-valued
            # "any_eq", e.g. comparing against a raw non-yes/no string).
            # Doing yes/no mapping before to_numeric matters: to_numeric
            # alone turns "yes"/"no" into NaN (silently dropping every
            # yes/no answer, always-False for any_gt/ge -- exactly as
            # wrong as the unconverted-string bug, just wrong in the
            # opposite direction), so a plain to_numeric coercion is not
            # sufficient by itself for yes/no-style questions.
            cols = pivoted[ids]
            if op != "any_eq" or isinstance(val, (int, float)):
                yes_no_map = {"yes": 1, "no": 0}
                cols = cols.apply(
                    lambda col: col.map(
                        lambda v: yes_no_map.get(v.strip().lower(), v) if isinstance(v, str) else v
                    )
                )
                cols = cols.apply(pd.to_numeric, errors="coerce")

            if op == "mean":
                score = cols.mean(axis=1)
                condition = score >= val
            elif op == "sum":
                score = cols.sum(axis=1)
                condition = score >= val
            elif op == "sum_le":
                score = cols.sum(axis=1)
                condition = score <= val
            elif op == "max":
                score = cols.max(axis=1)
                condition = score >= val
            elif op == "ge":
                condition = (cols >= val).any(axis=1)
            elif op == "any_eq":
                condition = (cols == val).any(axis=1)
            elif op == "any_gt":
                condition = (cols > val).any(axis=1)
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
    
    def __init__(self, likert_ids: list, shift_likert: bool = True, **kwargs) -> None:
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


class SleepAggregator(LabelAggregator):
    """Aggregates sleep duration survey responses into categorical classes.
    
    Morning surveys ask:
      - Question 54: What time did you fall asleep last night? (HH:MM am/pm)
      - Question 55: What time did you wake up this morning? (HH:MM am/pm)
      
    This aggregator:
      1. Parses time picker strings (e.g. "10:30 PM", "07:00 AM").
      2. Computes sleep duration in hours, accounting for overnight wraps.
      3. Classifies sleep hours:
         - < 5.0 hours -> Class 0
         - 5.0 to 10.0 hours -> Class 1
         - > 10.0 hours -> Class 2
    """
    
    requires_strings = True

    @property
    def num_classes(self) -> int:
        return 3

    def __init__(
        self,
        asleep_id: int = 54,
        awake_id: int = 55,
        short_sleep_threshold: float = 5.0,
        long_sleep_threshold: float = 10.0,
        **kwargs
    ) -> None:
        self.asleep_id = asleep_id
        self.awake_id = awake_id
        self.short_sleep_threshold = short_sleep_threshold
        self.long_sleep_threshold = long_sleep_threshold

    def get_question_ids(self) -> list:
        return [self.asleep_id, self.awake_id]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['survey_response_id', 'answer'])

        # Pivot to long format: columns are question_ids, index is survey_response_id
        pivoted = df.pivot(index='survey_response_id', columns='question_id', values='answer')
        
        # Ensure both question columns exist in the pivoted dataframe
        for qid in [self.asleep_id, self.awake_id]:
            if qid not in pivoted.columns:
                pivoted[qid] = None

        def calc_hours(row) -> Optional[float]:
            asleep = row[self.asleep_id]
            awake = row[self.awake_id]
            
            if pd.isna(asleep) or pd.isna(awake):
                return None
                
            try:
                def to_hours(t_str) -> Optional[float]:
                    if not isinstance(t_str, str):
                        return None
                    t_str = t_str.strip().upper()
                    dt = pd.to_datetime(t_str, format="%I:%M %p")
                    return dt.hour + dt.minute / 60.0

                asleep_h = to_hours(asleep)
                awake_h = to_hours(awake)
                
                if asleep_h is None or awake_h is None:
                    return None
                    
                if awake_h >= asleep_h:
                    return awake_h - asleep_h
                else:
                    return (awake_h - asleep_h) + 24.0
            except Exception:
                return None

        sleep_hours = pivoted.apply(calc_hours, axis=1)

        def categorize(hours) -> Optional[int]:
            if pd.isna(hours) or hours is None:
                return None
            if hours < self.short_sleep_threshold:
                return 0
            elif hours <= self.long_sleep_threshold:
                return 1
            else:
                return 2

        labels = sleep_hours.apply(categorize)
        
        res = pd.DataFrame({
            'survey_response_id': pivoted.index,
            'answer': labels
        }).dropna(subset=['answer'])
        
        res['answer'] = res['answer'].astype(int)
        return res.reset_index(drop=True)




