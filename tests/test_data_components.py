import numpy as np
import pandas as pd
import pytest
from src.data.components.cohort_splitter import CohortSplitter
from src.data.components.demographics_processor import DemographicsProcessor
from src.data.components.prev_prediction_linker import PrevPredictionLinker


@pytest.fixture
def dummy_splitter_data():
    return pd.DataFrame({
        "app_user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 4],
        "record_timestamp": pd.to_datetime([
            "2026-01-01 08:00:00",
            "2026-01-02 08:00:00",
            "2026-01-03 08:00:00",
            "2026-01-04 08:00:00",
            "2026-01-01 08:00:00",
            "2026-01-02 08:00:00",
            "2026-01-03 08:00:00",
            "2026-01-04 08:00:00",
            "2026-01-01 08:00:00",
            "2026-01-01 08:00:00",
        ]),
        "answer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_cohort_splitter_random(dummy_splitter_data):
    splitter = CohortSplitter(split_mode="random", train_val_test_split=(0.6, 0.2, 0.2), random_state=42)
    train_df, val_df, test_df = splitter.split(dummy_splitter_data)
    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2


def test_cohort_splitter_user(dummy_splitter_data):
    splitter = CohortSplitter(split_mode="user", train_val_test_split=(0.5, 0.25, 0.25), random_state=42)
    train_df, val_df, test_df = splitter.split(dummy_splitter_data)
    
    train_users = set(train_df["app_user_id"])
    val_users = set(val_df["app_user_id"])
    test_users = set(test_df["app_user_id"])

    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)


def test_cohort_splitter_longitudinal(dummy_splitter_data):
    splitter = CohortSplitter(split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25))
    train_df, val_df, test_df = splitter.split(dummy_splitter_data)

    # For user 1 (4 records): train gets 2, val gets 1, test gets 1
    u1_train = train_df[train_df["app_user_id"] == 1]
    u1_val = val_df[val_df["app_user_id"] == 1]
    u1_test = test_df[test_df["app_user_id"] == 1]

    assert len(u1_train) == 2
    assert len(u1_val) == 1
    assert len(u1_test) == 1
    assert u1_train["record_timestamp"].max() < u1_val["record_timestamp"].min()
    assert u1_val["record_timestamp"].max() < u1_test["record_timestamp"].min()


def test_demographics_processor():
    train_df = pd.DataFrame({"app_user_id": [1, 2]})
    master_df = pd.DataFrame({"app_user_id": [1, 2, 3]})
    demographics_df = pd.DataFrame({
        "app_user_id": [1, 2, 3],
        "gender": ["male", "female", "unknown"],
        "age": [20, 30, 40],
        "lgbt": ["no", "yes", "no"],
    })
    step_df = pd.DataFrame({
        "app_user_id": [1, 2],
        "app_source": ["iPhone 13", "health.connect.androidx"],
    })
    modality_dfs = {"step": step_df}

    processor = DemographicsProcessor(use_demographics=True)
    demo_map, default_demo = processor.fit_transform(
        train_df=train_df,
        demographics_df=demographics_df,
        modality_dfs=modality_dfs,
        master_df=master_df,
    )

    # Features should include: age_scaled, gender_male, gender_female, gender_unknown, lgbt_no, lgbt_yes, lgbt_unknown + 4 source channels
    # Total dim: 1 (age) + 3 (gender) + 3 (lgbt) + 4 (sources) = 11 dims
    assert processor.demographics_dim == 11
    assert len(default_demo) == 11

    # User 1 is iPhone -> iPhone channel set to 1.0 (sources categories: [android, iphone, apple_watch, unknown])
    # index order: gender_male (index 1), gender_female (index 2), gender_unknown (index 3)
    # sources: android (index 7), iphone (index 8), apple_watch (index 9), unknown (index 10)
    assert demo_map[1][8] == 1.0  # iPhone source active
    assert demo_map[2][7] == 1.0  # Android source active


def test_prev_prediction_linker(dummy_splitter_data):
    train_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 1].copy()
    val_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 2].copy()
    test_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 3].copy()

    t_df, v_df, test_res_df = PrevPredictionLinker.link(train_df, val_df, test_df)

    assert "prev_sample_idx" in t_df.columns
    assert "prev_sample_idx" in v_df.columns
    assert "prev_sample_idx" in test_res_df.columns

    # User 1 has 4 records, first should be -1, others point to previous index
    assert t_df.loc[0, "prev_sample_idx"] == -1
    assert t_df.loc[1, "prev_sample_idx"] == 0
    assert t_df.loc[2, "prev_sample_idx"] == 1
    assert t_df.loc[3, "prev_sample_idx"] == 2


def test_step_preprocessor():
    from src.data.components.preprocessing import StepPreprocessor
    
    # Create test data
    df = pd.DataFrame({
        "app_user_id": [22, 22, 26, 1],
        "start_timestamp": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00"
        ],
        "end_timestamp": [
            "2026-01-01 10:00:00", # 0 duration
            "2026-01-01 10:00:00", # 0 duration but steps = 0 (should keep, but android source drops it)
            "2026-01-01 10:05:00",
            "2026-01-01 10:05:00"
        ],
        "steps": [10, 0, 10, 10],
        "app_source": ["health.connect.android", "health.connect.android", "Health", "iPhone"]
    })
    
    preprocessor = StepPreprocessor()
    processed_df = preprocessor(df)
    
    # 1. 0 duration record with steps > 0 (user 22, row 0) should be dropped. 
    # Row 1 (steps=0) should be kept by duration check but dropped because app_user_id=22 contains android.
    # 2. iPhone source checks:
    # app_user_id 26: Health source mapped to Apple Watch.
    assert len(processed_df) == 2
    
    # Check remaining records
    processed_df = processed_df.reset_index(drop=True)
    assert processed_df.loc[0, "app_user_id"] == 26
    assert processed_df.loc[0, "app_source"] == "Apple Watch"
    assert processed_df.loc[1, "app_user_id"] == 1
    assert processed_df.loc[1, "app_source"] == "iPhone"


def test_calorie_preprocessor():
    from src.data.components.preprocessing import CaloriePreprocessor
    
    df = pd.DataFrame({
        "app_user_id": [1, 1, 1, 2, 2],
        "start_timestamp": [
            "2026-01-01 10:00:00", # study period (keep)
            "2026-01-01 10:00:00", # identical start/end
            "2025-08-01 10:00:00", # before study start (drop)
            "2026-01-01 10:00:00", # android conversion (>=3000 -> divide by 1000)
            "2026-01-01 10:00:00"  # duplicate of previous android record
        ],
        "end_timestamp": [
            "2026-01-01 10:05:00",
            "2026-01-01 10:00:00", # identical start/end (drop)
            "2025-08-01 10:05:00",
            "2026-01-01 10:05:00",
            "2026-01-01 10:05:00"
        ],
        "calories": [200.0, 100.0, 150.0, 3200.0, 3200.0],
        "app_source": ["iPhone", "iPhone", "iPhone", "health.connect.android", "health.connect.android"]
    })
    
    preprocessor = CaloriePreprocessor()
    processed_df = preprocessor(df)
    
    # Expected remaining rows:
    # 1. Row 0 (user 1, keep, calories = 200.0)
    # 2. Row 3 (user 2 android, keep and convert 3200.0 -> 3)
    # (Row 1 identical times dropped, Row 2 before study start dropped, Row 4 duplicate dropped)
    assert len(processed_df) == 2
    processed_df = processed_df.reset_index(drop=True)
    assert processed_df.loc[0, "app_user_id"] == 1
    assert processed_df.loc[0, "calories"] == 200.0
    assert processed_df.loc[1, "app_user_id"] == 2
    assert processed_df.loc[1, "calories"] == 3

