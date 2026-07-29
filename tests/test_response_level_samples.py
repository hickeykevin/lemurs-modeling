"""Tests for response-level sampling, sensor-coverage gating, and survey context."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import (
    BlockSampler,
    IntervalAwareSampler,
    LagSampler,
    OffsetSampler,
    RollingSampler,
)
from src.data.health_datamodule import HealthDataModule


USERS = [5, 6, 7, 8, 9, 10]


@pytest.fixture
def two_survey_data():
    """Morning+afternoon responses over two days, with matching step data.

    Six users, because user-level splitting needs enough participants to fill
    train, val and test without any of them coming out empty.
    """
    survey_rows, answer_rows, step_rows = [], [], []
    sid = 1
    for u in USERS:
        for day in ("2026-03-02", "2026-03-03"):
            for survey_id, hour in ((0, "09:00:00"), (1, "17:00:00")):
                survey_rows.append({
                    "id": sid, "app_user_id": u, "survey_id": survey_id,
                    "timestamp": pd.Timestamp(f"{day} {hour}"),
                })
                # Only user 5's first morning is positive, so that user has a
                # day whose two halves disagree.
                answer_rows.append({
                    "survey_response_id": sid, "question_id": 2,
                    "answer": 3 if sid == 1 else 0,
                })
                sid += 1
        for start, end in (
            ("2026-03-01 12:00:00", "2026-03-01 13:00:00"),
            ("2026-03-02 08:00:00", "2026-03-02 09:00:00"),
            ("2026-03-02 12:00:00", "2026-03-02 13:00:00"),
            ("2026-03-03 08:00:00", "2026-03-03 09:00:00"),
        ):
            step_rows.append({
                "app_user_id": u,
                "start_timestamp": pd.Timestamp(start),
                "end_timestamp": pd.Timestamp(end),
                "steps": 500,
            })

    return {
        "step": pd.DataFrame(step_rows),
        "survey_response": pd.DataFrame(survey_rows),
        "answer": pd.DataFrame(answer_rows),
    }


def _build(data, **kw):
    with patch("src.data.components.cohort_builder.DatabaseService") as M:
        db = M.return_value
        db.connect.return_value = True
        db.extract_from_database.side_effect = lambda t: data[t]
        builder = CohortBuilder(
            modalities=["step"],
            modality_cols={"step": "steps"},
            preprocessors=None,
            aggregator=MeanAggregator(question_ids=[2], threshold=1.0),
            exclude_user_ids=[],
            **kw,
        )
        return builder.build()


# ---------------------------------------------------------------------------
# Response-level labels
# ---------------------------------------------------------------------------

def test_collapse_none_keeps_every_response(two_survey_data):
    """Each survey response becomes its own sample."""
    _, master, _ = _build(two_survey_data, collapse_strategy="none")
    assert len(master) == 4 * len(USERS)
    assert set(master["survey_id"]) == {0, 1}


def test_collapse_daily_halves_the_cohort(two_survey_data):
    """The previous behaviour merges the two daily responses into one."""
    _, master, _ = _build(two_survey_data, collapse_strategy="max")
    assert len(master) == 2 * len(USERS)  # 2 days per user


def test_discordant_halves_survive_response_level(two_survey_data):
    """A positive morning and negative afternoon stay distinct."""
    _, master, _ = _build(two_survey_data, collapse_strategy="none")
    day = master[
        (master["app_user_id"] == 5)
        & (master["record_timestamp"].dt.date == pd.Timestamp("2026-03-02").date())
    ]
    assert sorted(day["answer"].tolist()) == [0, 1]

    # Collapsed, the negative half is overwritten by the positive one.
    _, collapsed, _ = _build(two_survey_data, collapse_strategy="max")
    day_c = collapsed[
        (collapsed["app_user_id"] == 5)
        & (collapsed["record_timestamp"].dt.date == pd.Timestamp("2026-03-02").date())
    ]
    assert len(day_c) == 1 and day_c["answer"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Referent window
# ---------------------------------------------------------------------------

def test_referent_hours_measures_gap_to_previous_response(two_survey_data):
    _, master, _ = _build(two_survey_data, collapse_strategy="none")
    u5 = master[master["app_user_id"] == 5].sort_values("record_timestamp")

    # First response of a user has no preceding prompt.
    assert pd.isna(u5["referent_hours"].iloc[0])
    # 09:00 -> 17:00 is 8h; 17:00 -> next 09:00 is 16h.
    np.testing.assert_allclose(u5["referent_hours"].iloc[1], 8.0)
    np.testing.assert_allclose(u5["referent_hours"].iloc[2], 16.0)


def test_is_morning_flags_the_morning_survey(two_survey_data):
    _, master, _ = _build(two_survey_data, collapse_strategy="none")
    assert (master.loc[master["survey_id"] == 0, "is_morning"] == 1.0).all()
    assert (master.loc[master["survey_id"] == 1, "is_morning"] == 0.0).all()


def test_referent_gap_does_not_cross_users(two_survey_data):
    _, master, _ = _build(two_survey_data, collapse_strategy="none")
    # The shift is grouped per user, so exactly one response per user — their
    # first — has no preceding prompt, rather than one across the whole cohort.
    assert master["referent_hours"].isna().sum() == master["app_user_id"].nunique()


def test_phq9_responses_excluded(two_survey_data):
    """The PHQ-9 runs on a different cadence and must not enter the cohort."""
    data = {k: v.copy() for k, v in two_survey_data.items()}
    data["survey_response"] = pd.concat([
        data["survey_response"],
        pd.DataFrame({"id": [99], "app_user_id": [5], "survey_id": [2],
                      "timestamp": pd.to_datetime(["2026-03-04 12:00:00"])}),
    ], ignore_index=True)
    data["answer"] = pd.concat([
        data["answer"],
        pd.DataFrame([{"survey_response_id": 99, "question_id": 2, "answer": 4}]),
    ], ignore_index=True)

    _, master, _ = _build(data, collapse_strategy="none")
    assert 99 not in set(master["survey_response_id"])
    assert 2 not in set(master["survey_id"])


# ---------------------------------------------------------------------------
# Sensor coverage gating
# ---------------------------------------------------------------------------

def test_window_bounds_reported_by_each_sensor_sampler():
    ts = pd.Timestamp("2026-04-30 12:00:00")
    for sampler in [
        RollingSampler(lookback_hours=24),
        OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
        BlockSampler(lookback_days=1),
        IntervalAwareSampler(),
    ]:
        bounds = sampler.window_bounds(ts)
        assert bounds is not None, type(sampler).__name__
        assert bounds[0] < bounds[1]


def test_lag_sampler_reports_no_window():
    """LagSampler reads no sensor data, so coverage gating must skip it."""
    assert LagSampler().window_bounds(pd.Timestamp("2026-04-30 12:00:00")) is None


def test_uncovered_samples_dropped(two_survey_data):
    """Responses whose window holds no records are removed before splitting."""
    data = {k: v.copy() for k, v in two_survey_data.items()}
    # Strip user 6's step data entirely; their responses lose all coverage.
    kept = USERS[:3]
    data["step"] = data["step"][data["step"]["app_user_id"].isin(kept)]

    with patch("src.data.components.cohort_builder.DatabaseService") as M:
        db = M.return_value
        db.connect.return_value = True
        db.extract_from_database.side_effect = lambda t: data[t]
        dm = HealthDataModule(
            exclude_user_ids=[],
            aggregator=MeanAggregator(question_ids=[2], threshold=1.0),
            sampler=IntervalAwareSampler(bin_edges_hours=[0, 6, 12]),
            collapse_strategy="none",
            train_val_test_split=(0.5, 0.25, 0.25),
            use_demographics=False,
            require_sensor_data=True,
        )
        dm.setup()

    assert set(dm.master_df["app_user_id"]) == set(kept)


def test_coverage_gate_disabled_keeps_everything(two_survey_data):
    data = {k: v.copy() for k, v in two_survey_data.items()}
    kept = USERS[:3]
    data["step"] = data["step"][data["step"]["app_user_id"].isin(kept)]

    with patch("src.data.components.cohort_builder.DatabaseService") as M:
        db = M.return_value
        db.connect.return_value = True
        db.extract_from_database.side_effect = lambda t: data[t]
        dm = HealthDataModule(
            exclude_user_ids=[],
            aggregator=MeanAggregator(question_ids=[2], threshold=1.0),
            sampler=IntervalAwareSampler(bin_edges_hours=[0, 6, 12]),
            collapse_strategy="none",
            train_val_test_split=(0.5, 0.25, 0.25),
            use_demographics=False,
            require_sensor_data=False,
        )
        dm.setup()

    assert set(dm.master_df["app_user_id"]) == set(USERS)


# ---------------------------------------------------------------------------
# Context features reach the model input
# ---------------------------------------------------------------------------

def test_survey_context_appended_to_demographics_vector(two_survey_data):
    with patch("src.data.components.cohort_builder.DatabaseService") as M:
        db = M.return_value
        db.connect.return_value = True
        db.extract_from_database.side_effect = lambda t: two_survey_data[t]
        dm = HealthDataModule(
            exclude_user_ids=[],
            aggregator=MeanAggregator(question_ids=[2], threshold=1.0),
            sampler=IntervalAwareSampler(bin_edges_hours=[0, 6, 12]),
            collapse_strategy="none",
            train_val_test_split=(0.5, 0.25, 0.25),
            use_demographics=False,
            use_survey_context=True,
        )
        dm.setup()

        sample = dm.data_train[0]
        context = sample[-1]
        # is_morning, referent_hours_scaled, referent_missing
        assert context.shape[-1] == dm.demographics_dim
        assert context.shape[-1] >= 3


def test_referent_scaling_uses_training_statistics_only(two_survey_data):
    with patch("src.data.components.cohort_builder.DatabaseService") as M:
        db = M.return_value
        db.connect.return_value = True
        db.extract_from_database.side_effect = lambda t: two_survey_data[t]
        dm = HealthDataModule(
            exclude_user_ids=[],
            aggregator=MeanAggregator(question_ids=[2], threshold=1.0),
            sampler=IntervalAwareSampler(bin_edges_hours=[0, 6, 12]),
            collapse_strategy="none",
            train_val_test_split=(0.5, 0.25, 0.25),
            use_demographics=False,
        )
        dm.setup()

    train_ref = pd.to_numeric(
        dm.data_train.data_links["referent_hours"], errors="coerce"
    ).dropna()
    np.testing.assert_allclose(dm.referent_mean, train_ref.mean(), rtol=1e-6)
