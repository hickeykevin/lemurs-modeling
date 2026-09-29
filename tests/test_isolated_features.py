import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.health_datamodule import HealthDataModule
from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import OffsetSampler


@pytest.fixture
def mock_health_db():
    step_df = pd.DataFrame({
        "app_user_id": [10, 11, 12, 13],
        "start_timestamp": pd.to_datetime([
            "2025-10-01 02:00:00", "2025-10-01 02:00:00",
            "2025-10-01 02:00:00", "2025-10-01 02:00:00"
        ]),
        "steps": [100, 150, 200, 250],
        "app_source": ["iPhone", "androidx", "Apple Watch", "unknown"],
    })
    distance_df = pd.DataFrame({
        "app_user_id": [10, 11, 12, 13],
        "start_timestamp": pd.to_datetime([
            "2025-10-01 02:00:00", "2025-10-01 02:00:00",
            "2025-10-01 02:00:00", "2025-10-01 02:00:00"
        ]),
        "distance": [80.0, 120.0, 160.0, 200.0],
        "app_source": ["iPhone", "androidx", "Apple Watch", "unknown"],
    })
    survey_df = pd.DataFrame({
        "id": [101, 102, 103, 104],
        "survey_id": [0, 1, 0, 1],  # morning, afternoon, morning, afternoon
        "app_user_id": [10, 11, 12, 13],
        "timestamp": pd.to_datetime([
            "2025-10-01 08:00:00", "2025-10-01 14:00:00",
            "2025-10-01 08:00:00", "2025-10-01 14:00:00"
        ]),
    })
    answer_df = pd.DataFrame([
        # user 10: 9 hours sleep
        {"survey_response_id": 101, "question_id": 54, "answer": "10:00 PM"},
        {"survey_response_id": 101, "question_id": 55, "answer": "07:00 AM"},
        {"survey_response_id": 101, "question_id": 2, "answer": "1"},

        # user 11: 4 hours sleep
        {"survey_response_id": 102, "question_id": 54, "answer": "12:00 AM"},
        {"survey_response_id": 102, "question_id": 55, "answer": "04:00 AM"},
        {"survey_response_id": 102, "question_id": 2, "answer": "0"},

        # user 12: missing sleep data
        {"survey_response_id": 103, "question_id": 2, "answer": "1"},

        # user 13: 11 hours sleep
        {"survey_response_id": 104, "question_id": 54, "answer": "09:00 PM"},
        {"survey_response_id": 104, "question_id": 55, "answer": "08:00 AM"},
        {"survey_response_id": 104, "question_id": 2, "answer": "0"},
    ])
    demo_df = pd.DataFrame([
        {"app_user_id": 10, "keyword": "age", "value": "25"},
        {"app_user_id": 11, "keyword": "age", "value": "30"},
        {"app_user_id": 12, "keyword": "age", "value": "35"},
        {"app_user_id": 13, "keyword": "age", "value": "40"},

        {"app_user_id": 10, "keyword": "gender identity", "value": "male"},
        {"app_user_id": 11, "keyword": "gender identity", "value": "female"},
        {"app_user_id": 12, "keyword": "gender identity", "value": "male"},
        {"app_user_id": 13, "keyword": "gender identity", "value": "female"},

        {"app_user_id": 10, "keyword": "lgbt", "value": "no"},
        {"app_user_id": 11, "keyword": "lgbt", "value": "yes"},
        {"app_user_id": 12, "keyword": "lgbt", "value": "no"},
        {"app_user_id": 13, "keyword": "lgbt", "value": "yes"},
    ])

    return {
        "step": step_df,
        "distance": distance_df,
        "survey_response": survey_df,
        "answer": answer_df,
        "demographic": demo_df,
    }


def make_dm(mock_tables, **kwargs):
    default_kwargs = {
        "aggregator": MeanAggregator([2]),
        "sampler": OffsetSampler(start_offset_hours=-6, end_offset_hours=0, resample_freq="1h", include_time_features=False),
        "modalities": ["step", "distance"],
        "split_mode": "longitudinal",
        "use_demographics": False,
        "use_age": False,
        "use_gender": False,
        "use_lgbt": False,
        "use_device_source": False,
        "use_sleep": False,
        "sleep_feature_mode": "both",
        "use_survey_context": False,
        "survey_context_mode": "both",
        "require_sensor_data": False,
        "exclude_user_ids": [],
    }
    default_kwargs.update(kwargs)

    with patch("src.data.components.cohort_builder.DatabaseService") as mock_db_class:
        mock_db = MagicMock()
        mock_db.connect.return_value = True
        mock_db.extract_from_database.side_effect = lambda table: mock_tables.get(table, pd.DataFrame()).copy()
        mock_db_class.return_value = mock_db

        dm = HealthDataModule(**default_kwargs)
        dm.setup()
        return dm


def test_condition_0_pure_sensor_baseline(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=False,
    )
    assert dm.demographics_dim == 0
    sample = dm.data_train[0]
    assert "features" in sample
    assert "demographics" not in sample


def test_condition_1_age_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=True,
        use_age=True,
        use_gender=False,
        use_lgbt=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=False,
    )
    assert dm.demographics_dim == 1
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (1,)


def test_condition_2_gender_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=True,
        use_age=False,
        use_gender=True,
        use_lgbt=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=False,
    )
    # categories: male, female + unknown = 3
    assert dm.demographics_dim == 3
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (3,)


def test_condition_3_lgbt_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=True,
        use_age=False,
        use_gender=False,
        use_lgbt=True,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=False,
    )
    # categories: no, yes + unknown = 3
    assert dm.demographics_dim == 3
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (3,)


def test_condition_4_all_demographics(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=True,
        use_age=True,
        use_gender=True,
        use_lgbt=True,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=False,
    )
    # 1 (age) + 3 (gender) + 3 (lgbt) = 7
    assert dm.demographics_dim == 7
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (7,)


def test_condition_5_device_source_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=True,
        use_survey_context=False,
        use_sleep=False,
    )
    # 4 device source features: [is_android, is_iphone, is_apple_watch, is_unknown]
    assert dm.demographics_dim == 4
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (4,)


def test_condition_6_survey_prompt_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=True,
        survey_context_mode="morning_only",
        use_sleep=False,
    )
    # 1 (is_morning)
    assert dm.demographics_dim == 1
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (1,)


def test_condition_7_referent_window_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=True,
        survey_context_mode="referent_only",
        use_sleep=False,
    )
    # 2 (referent_hours_scaled, referent_missing)
    assert dm.demographics_dim == 2
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (2,)


def test_condition_8_full_survey_context(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=True,
        survey_context_mode="both",
        use_sleep=False,
    )
    # 3 (is_morning, referent_hours_scaled, referent_missing)
    assert dm.demographics_dim == 3
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (3,)


def test_condition_9_sleep_hours_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=True,
        sleep_feature_mode="hours_only",
    )
    # 1 (sleep_hours_scaled)
    assert dm.demographics_dim == 1
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (1,)


def test_condition_10_sleep_category_only(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=True,
        sleep_feature_mode="category_only",
    )
    # 4 (sleep_class_0, 1, 2, sleep_unknown)
    assert dm.demographics_dim == 4
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (4,)


def test_condition_11_full_sleep(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=True,
        sleep_feature_mode="both",
    )
    # 5 (hours_scaled + class_0, 1, 2 + unknown)
    assert dm.demographics_dim == 5
    sample = dm.data_train[0]
    assert "demographics" in sample
    assert sample["demographics"].shape == (5,)


def test_sleep_feature_mode_none(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=False,
        use_sleep=True,
        sleep_feature_mode="none",
    )
    assert dm.demographics_dim == 0
    sample = dm.data_train[0]
    assert "demographics" not in sample


def test_survey_context_mode_none(mock_health_db):
    dm = make_dm(
        mock_health_db,
        use_demographics=False,
        use_device_source=False,
        use_survey_context=True,
        survey_context_mode="none",
        use_sleep=False,
    )
    assert dm.demographics_dim == 0
    sample = dm.data_train[0]
    assert "demographics" not in sample

