"""Excluded (test / discontinued) users are dropped from the label stream too."""

import pandas as pd

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.label_aggregators import RuleBasedAggregator


class _FakeDB:
    def __init__(self, tables):
        self.tables = tables

    def extract_from_database(self, name):
        return self.tables[name].copy()


def _builder(exclude):
    agg = RuleBasedAggregator(combination_logic="any", rules={"r": {"ids": [2], "op": "ge", "val": 2.0}})
    return CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=agg,
                         collapse_strategy="none", exclude_user_ids=exclude)


DB = _FakeDB({
    "survey_response": pd.DataFrame({
        "id": [1, 2, 3],
        "app_user_id": [5, 12, 1],
        "survey_id": [0, 0, 1],
        "timestamp": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:00", "2026-01-01 15:00"]),
    }),
    "answer": pd.DataFrame({"survey_response_id": [1, 2, 3], "question_id": [2, 2, 2], "answer": ["3", "1", "4"]}),
})


def test_excluded_users_have_no_labels():
    labels = _builder([1, 12])._extract_and_aggregate_labels(DB)
    assert set(labels.app_user_id) == {5}


def test_empty_exclusion_keeps_everyone():
    labels = _builder([])._extract_and_aggregate_labels(DB)
    assert set(labels.app_user_id) == {1, 5, 12}


def test_default_list_includes_test_accounts():
    assert {1, 2, 3, 12}.issubset(CohortBuilder.DEFAULT_EXCLUDED_USER_IDS)
