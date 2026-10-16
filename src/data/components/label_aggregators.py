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
