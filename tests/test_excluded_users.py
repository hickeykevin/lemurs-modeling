"""The default participant exclusion list is pinned: changing it changes every cohort and result."""

from src.data.components.cohort_builder import CohortBuilder


def test_default_excluded_user_ids():
    # Corrected 2026-10-07 (added 30, 53, 54, 55); the RQ1/RQ2 sweeps were rerun with this list.
    assert CohortBuilder.DEFAULT_EXCLUDED_USER_IDS == [1, 2, 3, 10, 12, 21, 22, 30, 43, 44, 53, 54, 55]
