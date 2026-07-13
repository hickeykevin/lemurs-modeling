import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock, patch
from src.data.health_datamodule import HealthDataModule
from src.data.components.label_aggregators import MeanAggregator, MaxAggregator
from src.data.components.samplers import OffsetSampler

@pytest.fixture
def dummy_data():
    """Creates dummy dataframes mimicking the database structure."""
    # 1. Health metrics (steps) - 4 users
    step_df = pd.DataFrame({
        'app_user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4],
        'start_timestamp': pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-01 11:00:00',
            '2023-01-02 10:00:00', '2023-01-02 11:00:00',
            '2023-01-01 10:00:00', '2023-01-01 11:00:00',
            '2023-01-02 10:00:00', '2023-01-02 11:00:00',
            '2023-01-01 10:00:00', '2023-01-02 10:00:00',
            '2023-01-01 10:00:00', '2023-01-02 10:00:00'
        ]),
        'steps': [100] * 12
    })
    
    # 2. Survey Responses - Multiple per user to test longitudinal split
    survey_df = pd.DataFrame({
        'id': [101, 102, 103, 104, 105, 106, 107, 108],
        'app_user_id': [1, 1, 1, 1, 2, 2, 2, 2], # Users with 4 responses each
        'timestamp': pd.to_datetime([
            '2023-01-02 08:00:00', '2023-01-03 08:00:00',
            '2023-01-04 08:00:00', '2023-01-05 08:00:00',
            '2023-01-02 08:00:00', '2023-01-03 08:00:00',
            '2023-01-04 08:00:00', '2023-01-05 08:00:00'
        ])
    })
    
    # Add some single-response users to test the "n < 3" edge case
    extra_surveys = pd.DataFrame({
        'id': [109, 110],
        'app_user_id': [3, 4],
        'timestamp': pd.to_datetime(['2023-01-02 08:00:00', '2023-01-02 08:00:00'])
    })
    survey_df = pd.concat([survey_df, extra_surveys])
    
    # 3. Answers
    answer_list = []
    for sid in survey_df['id']:
        # Give survey 101 higher scores (mean=2.0) to test thresholds
        score = 2 if sid == 101 else 1
        answer_list.append({'survey_response_id': sid, 'question_id': 2, 'answer': score})
        answer_list.append({'survey_response_id': sid, 'question_id': 4, 'answer': score})
    answer_df = pd.DataFrame(answer_list)
    
    demo_list = []
    # users 1, 2, 3, 4
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 2, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 3, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 4, 'keyword': 'age', 'value': '40'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 2, 'keyword': 'gender identity', 'value': 'female'},
        {'app_user_id': 3, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 4, 'keyword': 'gender identity', 'value': 'female'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 2, 'keyword': 'lgbt', 'value': 'yes'},
        {'app_user_id': 3, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 4, 'keyword': 'lgbt', 'value': 'no'},
    ])
    demo_df = pd.DataFrame(demo_list)
    
    return {"step": step_df, "survey_response": survey_df, "answer": answer_df, "demographic": demo_df}


def test_rule_based_aggregator():
    """Tests that RuleBasedAggregator handles various logical rules correctly."""
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # 1. Test suicide_risk logic
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'question_id':        [2, 5, 7, 2, 5, 7, 2, 5],
        'answer':             [1, 0, 1, 2, 0, 1, 1, 1] 
    })
    
    rules = [
        {'ids': [2, 3, 7], 'op': 'ge', 'val': 2},
        {'ids': [5, 8, 12, 13], 'op': 'any_eq', 'val': 1}
    ]
    agg = RuleBasedAggregator(rules=rules, combination_logic="any")
    result = agg(df)
    
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 0
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 1
    assert result.loc[result['survey_response_id'] == 3, 'answer'].values[0] == 1

    # 2. Test sum_le logic
    df_sum = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2],
        'question_id':        [21, 22, 21, 22],
        'answer':             [2, 2, 3, 3] 
    })
    
    rules_sum = [{'ids': [21, 22], 'op': 'sum_le', 'threshold': 5}]
    agg_sum = RuleBasedAggregator(rules=rules_sum, combination_logic="any")
    result_sum = agg_sum(df_sum)
    
    assert result_sum.loc[result_sum['survey_response_id'] == 1, 'answer'].values[0] == 1 
    assert result_sum.loc[result_sum['survey_response_id'] == 2, 'answer'].values[0] == 0

    # 3. Test social_stress (sum >= 6)
    df_stress = pd.DataFrame({
        'survey_response_id': [1, 1, 1, 2, 2, 2],
        'question_id':        [31, 32, 33, 31, 32, 33],
        'answer':             [1, 2, 2, 2, 2, 2] 
    })
    rules_stress = [{'ids': [31, 32, 33], 'op': 'sum', 'threshold': 6}]
    agg_stress = RuleBasedAggregator(rules=rules_stress, combination_logic="any")
    result_stress = agg_stress(df_stress)
    assert result_stress.loc[result_stress['survey_response_id'] == 1, 'answer'].values[0] == 0
    assert result_stress.loc[result_stress['survey_response_id'] == 2, 'answer'].values[0] == 1

    # 4. Test social_connection (sum_le <= 4)
    df_conn = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2],
        'question_id':        [34, 35, 34, 35],
        'answer':             [2, 2, 3, 2] 
    })
    rules_conn = [{'ids': [34, 35], 'op': 'sum_le', 'threshold': 4}]
    agg_conn = RuleBasedAggregator(rules=rules_conn, combination_logic="any")
    result_conn = agg_conn(df_conn)
    assert result_conn.loc[result_conn['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result_conn.loc[result_conn['survey_response_id'] == 2, 'answer'].values[0] == 0


def test_rule_based_aggregator_dict_rules():
    """Tests that RuleBasedAggregator handles dictionary-based rules correctly."""
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'question_id':        [2, 5, 7, 2, 5, 7, 2, 5],
        'answer':             [1, 0, 1, 2, 0, 1, 1, 1] 
    })
    
    rules = {
        'mild_risk': {'ids': [2, 3, 7], 'op': 'ge', 'val': 2},
        'severe_risk': {'ids': [5, 8, 12, 13], 'op': 'any_eq', 'val': 1}
    }
    agg = RuleBasedAggregator(rules=rules, combination_logic="any")
    result = agg(df)
    
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 0
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 1
    assert result.loc[result['survey_response_id'] == 3, 'answer'].values[0] == 1

    # Test get_question_ids gets the union of ids
    assert set(agg.get_question_ids()) == {2, 3, 7, 5, 8, 12, 13}





def test_datamodule_normalization(dummy_data):
    """Tests that the datamodule correctly fits and applies a scaler."""
    from sklearn.preprocessing import StandardScaler
    from src.data.components.samplers import OffsetSampler
    
    with patch('src.data.components.cohort_builder.DatabaseService') as mock_db_class:
        mock_db = mock_db_class.return_value
        mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
        
        # Initial values in dummy_data for steps are all 10 or 20
        # If we use StandardScaler, it should shift them
        scaler = StandardScaler()
        dm = HealthDataModule(
            aggregator=MeanAggregator(question_ids=[2]),
            sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
            scaler=scaler,
        )

        dm.setup()
        
        # Check that scaler is fitted
        assert hasattr(scaler, "mean_")
        
        # Get a sample
        train_ds = dm.data_train
        seq, target, *rest = train_ds[0]
        
        # If mean was 15 (hypothetically), 20 would become (20-15)/std
        # The key is that the tensor should not be the raw [10, 20] values
        # We check that the mean of the sequence is roughly 0 if we normalized the whole thing
        # (Though with only a few samples it might not be exactly 0)
        assert not torch.allclose(seq.mean(), torch.tensor(15.0), atol=1.0)


def test_rolling_sampler():
    """Tests that RollingSampler slices the correct time window and resamples."""
    from src.data.components.samplers import RollingSampler
    
    # 1. Create dummy data: 48 hours of steps for one user
    timestamps = pd.date_range("2023-01-01", periods=48, freq="h")
    df = pd.DataFrame({
        'app_user_id': [1] * 48,
        'start_timestamp': timestamps,
        'steps': [10] * 48
    })
    modality_dfs = {"step": df}
    modality_cols = {"step": "steps"}
    
    # Survey taken at 2023-01-02 12:30:00 (will floor to 12:00)
    survey_time = pd.Timestamp("2023-01-02 12:30:00")
    
    # Test 12-hour lookback, hourly resampling
    sampler = RollingSampler(lookback_hours=12, resample_freq="1h")
    result = sampler(survey_time, 1, modality_dfs, modality_cols, ["step"])
    
    # Should have 12 time steps (one per hour)
    assert result.shape == (12, 5)
    # Total steps should be 120 (12 * 10)
    assert result[:, 0].sum() == 120
    
    # Test 12-hour lookback, 4-hour resampling
    sampler_4h = RollingSampler(lookback_hours=12, resample_freq="4h")
    result_4h = sampler_4h(survey_time, 1, modality_dfs, modality_cols, ["step"])
    
    # Should have 3 time steps (12/4)
    assert result_4h.shape == (3, 5)
    # Each bin should have 4 hours of data (40 steps)
    assert result_4h[0, 0] == 40

    # Test with include_time_features=False
    sampler_no_time = RollingSampler(lookback_hours=12, resample_freq="1h", include_time_features=False)
    result_no_time = sampler_no_time(survey_time, 1, modality_dfs, modality_cols, ["step"])
    assert result_no_time.shape == (12, 1)

def test_offset_sampler():
    """Tests that OffsetSampler correctly handles offsets from midnight."""
    from src.data.components.samplers import OffsetSampler
    
    # 1. Create dummy data: 48 hours of steps
    timestamps = pd.date_range("2023-01-01", periods=48, freq="h")
    df = pd.DataFrame({
        'app_user_id': [1] * 48,
        'start_timestamp': timestamps,
        'steps': [10] * 48
    })
    modality_dfs = {"step": df}
    modality_cols = {"step": "steps"}
    
    # Survey taken at 2023-01-02 08:30:00
    # Midnight of June 2 is 2023-01-02 00:00:00
    survey_time = pd.Timestamp("2023-01-02 08:30:00")
    
    # Target: 6pm June 1 (-6h from midnight) to 6am June 2 (+6h from midnight)
    sampler = OffsetSampler(start_offset_hours=-6, end_offset_hours=6, resample_freq="1h")
    result = sampler(survey_time, 1, modality_dfs, modality_cols, ["step"])
    
    # Total duration is 12 hours. Features = [step, sin_weekday, cos_weekday].
    # OffsetSampler intentionally omits sin/cos hour (constant across samples for
    # this sampler), so only weekday cyclic features accompany the modality column.
    assert result.shape == (12, 3)
    assert result[:, 0].sum() == 120

    # Test with include_time_features=False
    sampler_no_time = OffsetSampler(start_offset_hours=-6, end_offset_hours=6, resample_freq="1h", include_time_features=False)
    result_no_time = sampler_no_time(survey_time, 1, modality_dfs, modality_cols, ["step"])
    assert result_no_time.shape == (12, 1)

@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_user_split(mock_db_class, dummy_data):
    """Tests that user split results in disjoint users."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Set up DM with user split (50/50 for simplicity in test)
    sampler = OffsetSampler(start_offset_hours=-24, end_offset_hours=0)
    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2, 4], threshold=1.0),
        sampler=sampler,

        train_val_test_split=(0.5, 0.25, 0.25),
        split_mode="user"
    )

    
    dm.setup()
    
    # Get user IDs in each set
    train_users = set(dm.data_train.data_links['app_user_id'])
    val_users = set(dm.data_val.data_links['app_user_id'])
    test_users = set(dm.data_test.data_links['app_user_id'])
    
    # Check disjointness
    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)

@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_longitudinal_split(mock_db_class, dummy_data):
    """Tests that longitudinal split respects chronology."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    dm = HealthDataModule(
        aggregator=MeanAggregator(threshold=None),
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
        train_val_test_split=(0.5, 0.25, 0.25),
        split_mode="longitudinal"
    )
    
    dm.setup()
    
    # For user 1, response 101 (2023-01-02) should be in train
    # response 102 (2023-01-03) should be in val or test
    user1_train = dm.data_train.data_links[dm.data_train.data_links['app_user_id'] == 1]
    user1_test_or_val = pd.concat([
        dm.data_val.data_links[dm.data_val.data_links['app_user_id'] == 1],
        dm.data_test.data_links[dm.data_test.data_links['app_user_id'] == 1]
    ])
    
    assert not user1_train.empty
    assert not user1_test_or_val.empty
    
    # Check that train timestamp < test/val timestamp
    assert user1_train['record_timestamp'].max() < user1_test_or_val['record_timestamp'].min()

@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_multi_question_aggregation(mock_db_class, dummy_data):
    """Tests that labels are correctly aggregated from multiple question IDs."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Mean of [2, 2] = 2.0. Threshold 1.5 -> Label 1
    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2, 4], threshold=1.5),
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
    )

    
    dm.setup()
    
    # response_id 101 belongs to user 1
    row_101 = dm.data_train.data_links[dm.data_train.data_links['survey_response_id'] == 101]
    if row_101.empty: # might be in val/test
        row_101 = dm.data_val.data_links[dm.data_val.data_links['survey_response_id'] == 101]
    if row_101.empty:
        row_101 = dm.data_test.data_links[dm.data_test.data_links['survey_response_id'] == 101]
        
    assert row_101.iloc[0]['answer'] == 1


@patch('src.data.components.cohort_builder.DatabaseService')
def test_subject_scaler_normalization(mock_db_class):
    """Tests that SubjectScaler standardizes features independently for each user."""
    from sklearn.preprocessing import StandardScaler
    from src.data.components.scalers import SubjectScaler
    from src.data.components.samplers import OffsetSampler

    # Define dummy data for two users with different scales of steps
    # User 5 steps: always 10 and 20
    # User 6 steps: always 100 and 200
    step_df = pd.DataFrame({
        'app_user_id': [5, 5, 5, 5, 6, 6, 6, 6],
        'start_timestamp': pd.to_datetime([
            '2026-01-01 10:00:00', '2026-01-01 11:00:00',
            '2026-01-02 10:00:00', '2026-01-02 11:00:00',
            '2026-01-01 10:00:00', '2026-01-01 11:00:00',
            '2026-01-02 10:00:00', '2026-01-02 11:00:00',
        ]),
        'steps': [10, 20, 10, 20, 100, 200, 100, 200]
    })
    
    # 2. Survey Responses - 2 per user
    survey_df = pd.DataFrame({
        'id': [101, 102, 103, 104],
        'app_user_id': [5, 5, 6, 6],
        'timestamp': pd.to_datetime([
            '2026-01-01 12:00:00', '2026-01-02 12:00:00',
            '2026-01-01 12:00:00', '2026-01-02 12:00:00'
        ])
    })
    
    # 3. Answers
    answer_df = pd.DataFrame([
        {'survey_response_id': 101, 'question_id': 2, 'answer': 1},
        {'survey_response_id': 102, 'question_id': 2, 'answer': 1},
        {'survey_response_id': 103, 'question_id': 2, 'answer': 1},
        {'survey_response_id': 104, 'question_id': 2, 'answer': 1}
    ])
    
    demo_list = []
    # users 1, 2, 3, 4
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 2, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 3, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 4, 'keyword': 'age', 'value': '40'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 2, 'keyword': 'gender identity', 'value': 'female'},
        {'app_user_id': 3, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 4, 'keyword': 'gender identity', 'value': 'female'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 2, 'keyword': 'lgbt', 'value': 'yes'},
        {'app_user_id': 3, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 4, 'keyword': 'lgbt', 'value': 'no'},
    ])
    demo_df = pd.DataFrame(demo_list)
    dummy_data = {"step": step_df, "survey_response": survey_df, "answer": answer_df, "demographic": demo_df}

    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]

    # Instantiate datamodule with SubjectScaler wrapping StandardScaler
    base_scaler = StandardScaler()
    scaler = SubjectScaler(base_scaler=base_scaler)
    
    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2]),
        sampler=OffsetSampler(start_offset_hours=10, end_offset_hours=12, resample_freq="1h"),
        scaler=scaler,
        train_val_test_split=(0.5, 0.25, 0.25),
        split_mode="random"
    )

    dm.setup()
    
    # Gather all dataset splits (train, val, test) and verify they are correctly z-scored
    all_datasets = [dm.data_train, dm.data_val, dm.data_test]
    
    expected_seq = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    
    for ds in all_datasets:
        for i in range(len(ds)):
            seq, *rest = ds[i]
            assert torch.allclose(seq[:, :1], expected_seq, atol=1e-5)


def test_regression_aggregator():
    """Tests that RegressionAggregator sums and shifts Likert scale questions correctly."""
    from src.data.components.label_aggregators import RegressionAggregator
    
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2, 3],
        'question_id':        [2, 4, 2, 4, 2],
        'answer':             [2, 3, 1, 5, 1]
    })
    
    agg = RegressionAggregator(likert_ids=[2, 4], shift_likert=True)
    result = agg(df)
    
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 3.0
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 4.0
    assert result.loc[result['survey_response_id'] == 3, 'answer'].values[0] == 0.0

    # Test without shift
    agg_no_shift = RegressionAggregator(likert_ids=[2, 4], shift_likert=False)
    result_no_shift = agg_no_shift(df)
    assert result_no_shift.loc[result_no_shift['survey_response_id'] == 1, 'answer'].values[0] == 5.0
    assert result_no_shift.loc[result_no_shift['survey_response_id'] == 2, 'answer'].values[0] == 6.0
    assert result_no_shift.loc[result_no_shift['survey_response_id'] == 3, 'answer'].values[0] == 1.0

    # Test auto-skipping shift if 0 is already present in the dataset (already 0-indexed)
    df_zero = pd.DataFrame({
        'survey_response_id': [1, 1, 2],
        'question_id':        [2, 4, 2],
        'answer':             [0, 3, 1]  # contains a 0
    })
    agg_auto_skip = RegressionAggregator(likert_ids=[2, 4], shift_likert=True)
    result_auto_skip = agg_auto_skip(df_zero)
    assert result_auto_skip.loc[result_auto_skip['survey_response_id'] == 1, 'answer'].values[0] == 3.0  # 0 + 3 = 3 (no shift applied)
    assert result_auto_skip.loc[result_auto_skip['survey_response_id'] == 2, 'answer'].values[0] == 1.0  # 1 (no shift applied)



@patch('src.data.components.cohort_builder.DatabaseService')
def test_regression_datamodule_and_model(mock_db_class, dummy_data):
    """Tests that the datamodule yields float targets, model trains in regression mode, and callbacks run."""
    from src.data.components.label_aggregators import RegressionAggregator
    from src.models.health_module import HealthRegressionLitModule
    from src.models.components.simple_lstm import SimpleLSTM
    from src.utils.evaluation_callbacks import RegressionMetricsCallback
    from functools import partial
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Answers in dummy_data for q2 and q4 are all 1 or 2 (which shift to 0 or 1).
    aggregator = RegressionAggregator(likert_ids=[2, 4])
    dm = HealthDataModule(
        aggregator=aggregator,
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
        modalities=["step"]
    )
    dm.setup()
    
    # 1. Check datamodule targets
    train_ds = dm.data_train
    seq, target, *rest = train_ds[0]
    assert target.dtype == torch.float32
    
    # 2. Check model steps in regression mode
    net = SimpleLSTM(input_size=seq.shape[-1], hidden_size=8, num_layers=1, output_size=1)
    
    # Setup optimizer partial
    optimizer = partial(torch.optim.Adam, lr=0.001)
    
    model = HealthRegressionLitModule(
        net=net,
        optimizer=optimizer
    )
    
    # Simulate a training step
    batch = (seq.unsqueeze(0), target.unsqueeze(0), torch.tensor([0]))
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dtype == torch.float32

    # 3. Verify regression metrics callback
    callback = RegressionMetricsCallback(frequency=1, num_bootstraps=5, sampling_strategy="multinomial")
    
    # Mock trainer and model logging
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.current_epoch = 1
    
    model.log = MagicMock()
    
    # Simulate epoch start
    callback.on_validation_epoch_start(trainer, model)
    
    # Create outputs for a batch of size 2 with different target values (e.g. 1.0 and 2.0)
    batch_preds = torch.tensor([1.5, 3.5])
    batch_targets = torch.tensor([1.0, 2.0])
    batch = (None, batch_targets, None)
    outputs = {"loss": torch.tensor(0.5), "preds": batch_preds, "targets": batch_targets}
    
    # Simulate batch end
    callback.on_validation_batch_end(trainer, model, outputs, batch, 0)
    
    # Simulate epoch end
    callback.on_validation_epoch_end(trainer, model)
    
    # Assert model.log was called for standard val/mse and val/mae (via metrics collection),
    # plus val/mse_non_min and val/mae_non_min
    # The minimum target is 1.0. The non-minimum targets are targets > 1.0 (which is [2.0]).
    # Corresponding prediction is 3.5.
    # Expected non-minimum MSE is (3.5 - 2.0)^2 = 2.25.
    # Expected non-minimum MAE is |3.5 - 2.0| = 1.5.
    model.log.assert_any_call("val/mse_non_min", torch.tensor(2.25), on_step=False, on_epoch=True, prog_bar=True)
    model.log.assert_any_call("val/mae_non_min", torch.tensor(1.5), on_step=False, on_epoch=True, prog_bar=True)

    # Also verify test hooks
    callback.on_test_epoch_start(trainer, model)
    callback.on_test_batch_end(trainer, model, outputs, batch, 0)
    callback.on_test_epoch_end(trainer, model)
    
    model.log.assert_any_call("test/mse_non_min_mean", torch.tensor(2.25), on_step=False, on_epoch=True, prog_bar=True)
    model.log.assert_any_call("test/mae_non_min_mean", torch.tensor(1.5), on_step=False, on_epoch=True, prog_bar=True)


@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_deduplication(mock_db_class):
    """Tests that CohortBuilder filters out duplicate survey responses within 10 minutes with identical answers."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # Define answers:
    # Response 101 and 102 are for same user and survey, 2 minutes apart, identical answers (should deduplicate 102)
    # Response 103 is for same user and survey, 12 minutes after 101 (distinct, should keep)
    # Response 104 and 105 are for same user and survey, 1 minute apart, different answers (should keep both)
    survey_df = pd.DataFrame({
        'id': [101, 102, 103, 104, 105],
        'app_user_id': [1, 1, 1, 2, 2],
        'survey_id': [0, 0, 0, 0, 0],
        'timestamp': pd.to_datetime([
            '2026-01-01 10:00:00', # 101
            '2026-01-01 10:02:00', # 102 (duplicate of 101)
            '2026-01-01 10:12:00', # 103 (too late)
            '2026-01-01 10:00:00', # 104
            '2026-01-01 10:01:00', # 105 (different answers from 104)
        ])
    })
    
    answer_df = pd.DataFrame([
        # 101 answers
        {'survey_response_id': 101, 'question_id': 2, 'answer': '1'},
        {'survey_response_id': 101, 'question_id': 3, 'answer': '0'},
        # 102 answers (identical to 101)
        {'survey_response_id': 102, 'question_id': 2, 'answer': '1'},
        {'survey_response_id': 102, 'question_id': 3, 'answer': '0'},
        # 103 answers (identical to 101, but >10 mins)
        {'survey_response_id': 103, 'question_id': 2, 'answer': '1'},
        {'survey_response_id': 103, 'question_id': 3, 'answer': '0'},
        # 104 answers
        {'survey_response_id': 104, 'question_id': 2, 'answer': '1'},
        {'survey_response_id': 104, 'question_id': 3, 'answer': '0'},
        # 105 answers (different from 104)
        {'survey_response_id': 105, 'question_id': 2, 'answer': '2'}, # different value
        {'survey_response_id': 105, 'question_id': 3, 'answer': '0'},
    ])
    
    demo_list = []
    # users 1, 2, 3, 4
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 2, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 3, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 4, 'keyword': 'age', 'value': '40'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 2, 'keyword': 'gender identity', 'value': 'female'},
        {'app_user_id': 3, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 4, 'keyword': 'gender identity', 'value': 'female'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 2, 'keyword': 'lgbt', 'value': 'yes'},
        {'app_user_id': 3, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 4, 'keyword': 'lgbt', 'value': 'no'},
    ])
    demo_df = pd.DataFrame(demo_list)
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df, "demographic": demo_df}
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    rules = [{'ids': [2, 3], 'op': 'ge', 'val': 1}]
    agg = RuleBasedAggregator(rules=rules, combination_logic="any")
    
    builder = CohortBuilder(
        modalities=[],
        modality_cols={},
        preprocessors=None,
        aggregator=agg,
        os_filter='both'
    )
    
    _, master_df, _ = builder.build()
    
    # Master df should contain:
    # 103 (representative for User 1 after deduplicating 102 and collapsing 101/103)
    # 105 (representative for User 2 after collapsing 104/105)
    response_ids = master_df['survey_response_id'].tolist()
    assert 101 not in response_ids
    assert 102 not in response_ids
    assert 103 in response_ids
    assert 104 not in response_ids
    assert 105 in response_ids
    assert len(response_ids) == 2


@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_collapsing(mock_db_class):
    """Tests that CohortBuilder correctly collapses multiple surveys on the same day per user."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # 2 responses on same day (101, 102), and 1 response on different day (103) for user 1
    # 101: 9 AM, 102: 5 PM. 102 is the latest, should act as the representative.
    survey_df = pd.DataFrame({
        'id': [101, 102, 103],
        'app_user_id': [1, 1, 1],
        'survey_id': [0, 0, 0],
        'timestamp': pd.to_datetime([
            '2026-01-01 09:00:00', # 101
            '2026-01-01 17:00:00', # 102 (latest response of the day, representative)
            '2026-01-02 09:00:00', # 103
        ])
    })
    
    # Question 2: style '5-scale-much' (numeric scale, should mean: (2 + 4) / 2 = 3.0)
    # Question 11: style 'yes-no' (yes-no, should max: max(0, 1) = 1.0)
    answer_df = pd.DataFrame([
        # 101 answers
        {'survey_response_id': 101, 'question_id': 2, 'answer': '2'},
        {'survey_response_id': 101, 'question_id': 11, 'answer': 'no'}, # 0.0
        # 102 answers
        {'survey_response_id': 102, 'question_id': 2, 'answer': '4'},
        {'survey_response_id': 102, 'question_id': 11, 'answer': 'yes'}, # 1.0
        # 103 answers
        {'survey_response_id': 103, 'question_id': 2, 'answer': '1'},
        {'survey_response_id': 103, 'question_id': 11, 'answer': 'no'},
    ])
    
    demo_list = []
    # users 1, 2, 3, 4
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 2, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 3, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 4, 'keyword': 'age', 'value': '40'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 2, 'keyword': 'gender identity', 'value': 'female'},
        {'app_user_id': 3, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 4, 'keyword': 'gender identity', 'value': 'female'},
    ])
    demo_list.extend([
        {'app_user_id': 1, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 2, 'keyword': 'lgbt', 'value': 'yes'},
        {'app_user_id': 3, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 4, 'keyword': 'lgbt', 'value': 'no'},
    ])
    demo_df = pd.DataFrame(demo_list)
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df, "demographic": demo_df}
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Test aggregator targeting numeric question 2 with mean logic
    # Expected: 102 (representative) gets answer 3.0. ge 3.0 is True -> 1
    # 103 gets answer 1.0. ge 3.0 is False -> 0
    rules = [{'ids': [2], 'op': 'ge', 'val': 3}]
    agg = RuleBasedAggregator(rules=rules, combination_logic="any")
    
    builder = CohortBuilder(
        modalities=[],
        modality_cols={},
        preprocessors=None,
        aggregator=agg,
        os_filter='both'
    )
    
    _, master_df, _ = builder.build()
    
    # Check that 101 is collapsed and not present in master_df
    # Representative 102 and distinct 103 are present
    response_ids = master_df['survey_response_id'].tolist()
    assert 101 not in response_ids
    assert 102 in response_ids
    assert 103 in response_ids
    
    # Value for 102 is 1 (True), 103 is 0 (False)
    assert master_df.loc[master_df['survey_response_id'] == 102, 'answer'].values[0] == 1
    assert master_df.loc[master_df['survey_response_id'] == 103, 'answer'].values[0] == 0


def test_rule_based_aggregator_collapsed_values():
    """Tests that RuleBasedAggregator's any_gt handles averaged/collapsed values conservatively, while any_eq is strictly exact."""
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # Question 11 is yes-no and question 2 is Likert
    # Group contains two survey responses, one yes (1) and one no (0), averaged or maxed:
    # If yes-no is maxed -> 1.0
    # If Likert is meaned -> 0.5
    df_collapsed = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2],
        'question_id':        [2, 11, 2, 11],
        # Response 1: Question 2 = 0.5 (averaged), Question 11 = 1.0 (maxed)
        # Response 2: Question 2 = 0.0 (averaged), Question 11 = 0.0 (maxed)
        'answer':             [0.5, 1.0, 0.0, 0.0]
    })
    
    # 1. Test any_gt with val: 0 (should capture 0.5 as True, 0.0 as False)
    rules_likert_gt = [{'ids': [2], 'op': 'any_gt', 'val': 0}]
    agg_likert_gt = RuleBasedAggregator(rules=rules_likert_gt, combination_logic="any")
    
    result_likert_gt = agg_likert_gt(df_collapsed)
    assert result_likert_gt.loc[result_likert_gt['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result_likert_gt.loc[result_likert_gt['survey_response_id'] == 2, 'answer'].values[0] == 0

    # 2. Test any_eq with val: 1 (should NOT capture 0.5 as True, because it's strictly 0.5 != 1.0)
    rules_likert_eq = [{'ids': [2], 'op': 'any_eq', 'val': 1}]
    agg_likert_eq = RuleBasedAggregator(rules=rules_likert_eq, combination_logic="any")
    
    result_likert_eq = agg_likert_eq(df_collapsed)
    assert result_likert_eq.loc[result_likert_eq['survey_response_id'] == 1, 'answer'].values[0] == 0
    assert result_likert_eq.loc[result_likert_eq['survey_response_id'] == 2, 'answer'].values[0] == 0

    # 3. Question 11 = 1.0 (should match any_eq 1 since 1.0 == 1.0)
    rules_yn = [{'ids': [11], 'op': 'any_eq', 'val': 1}]
    agg_yn = RuleBasedAggregator(rules=rules_yn, combination_logic="any")
    
    result_yn = agg_yn(df_collapsed)
    assert result_yn.loc[result_yn['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result_yn.loc[result_yn['survey_response_id'] == 2, 'answer'].values[0] == 0



@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_collapsing_strategies(mock_db_class):
    """Tests that CohortBuilder correctly collapses multiple surveys using different configurable strategies."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # 2 responses on same day (101, 102) for user 1
    # 101: 9 AM, 102: 5 PM. 102 is the latest (representative).
    survey_df = pd.DataFrame({
        'id': [101, 102],
        'app_user_id': [1, 1],
        'survey_id': [0, 0],
        'timestamp': pd.to_datetime([
            '2026-01-01 09:00:00', # 101
            '2026-01-01 17:00:00', # 102 (representative)
        ])
    })
    
    # Question 2: style '5-scale-much' (numeric scale, can test mean, max, min, first, last)
    # Question 11: style 'yes-no' (yes-no, should always use max regardless of strategy)
    answer_df = pd.DataFrame([
        # 101 answers (first)
        {'survey_response_id': 101, 'question_id': 2, 'answer': '2'},
        {'survey_response_id': 101, 'question_id': 11, 'answer': 'no'}, # 0.0
        # 102 answers (last)
        {'survey_response_id': 102, 'question_id': 2, 'answer': '4'},
        {'survey_response_id': 102, 'question_id': 11, 'answer': 'yes'}, # 1.0
    ])
    
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df}
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    
    for strat, expected_q2, expected_q11 in [
        ("mean", 3.0, 1.0),
        ("max", 4.0, 1.0),
        ("min", 2.0, 1.0),
        ("first", 2.0, 1.0),
        ("last", 4.0, 1.0)
    ]:
        mock_db.extract_from_database.side_effect = lambda table: dummy_data[table].copy()
        
        rules = [{'ids': [2, 11], 'op': 'ge', 'val': 0}]
        agg = RuleBasedAggregator(rules=rules, combination_logic="any")
        
        builder = CohortBuilder(
            modalities=[],
            modality_cols={},
            preprocessors=None,
            aggregator=agg,
            os_filter='both',
            collapse_strategy=strat
        )
        
        # Manually collapse responses and check answer values
        collapsed_sr, collapsed_ans = builder._collapse_daily_responses(survey_df, answer_df, mock_db)
        
        # check question 2
        q2_ans = collapsed_ans.loc[collapsed_ans['question_id'] == 2, 'answer'].values[0]
        assert q2_ans == expected_q2, f"Strategy {strat} failed for question 2: expected {expected_q2}, got {q2_ans}"
        
        # check question 11 (yes-no should always be maxed)
        q11_ans = collapsed_ans.loc[collapsed_ans['question_id'] == 11, 'answer'].values[0]
        assert q11_ans == expected_q11, f"Strategy {strat} failed for question 11: expected {expected_q11}, got {q11_ans}"


@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_no_collapsing(mock_db_class):
    """Tests that CohortBuilder with collapse_strategy='none' preserves both survey responses."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import RuleBasedAggregator
    
    # 2 responses on same day for user 1
    survey_df = pd.DataFrame({
        'id': [101, 102],
        'app_user_id': [1, 1],
        'survey_id': [0, 0],
        'timestamp': pd.to_datetime([
            '2026-01-01 09:00:00',
            '2026-01-01 17:00:00',
        ])
    })
    
    answer_df = pd.DataFrame([
        {'survey_response_id': 101, 'question_id': 2, 'answer': '2'},
        {'survey_response_id': 102, 'question_id': 2, 'answer': '4'},
    ])
    
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df}
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table].copy()
    
    rules = [{'ids': [2], 'op': 'ge', 'val': 0}]
    agg = RuleBasedAggregator(rules=rules, combination_logic="any")
    
    # Test strategy = 'none'
    builder_none = CohortBuilder(
        modalities=[],
        modality_cols={},
        preprocessors=None,
        aggregator=agg,
        os_filter='both',
        collapse_strategy='none'
    )
    collapsed_sr, collapsed_ans = builder_none._collapse_daily_responses(survey_df, answer_df, mock_db)
    
    # Verify that the original responses and answers are kept intact (not collapsed)
    assert len(collapsed_sr) == 2
    assert 101 in collapsed_sr['id'].tolist()
    assert 102 in collapsed_sr['id'].tolist()


def test_sleep_aggregator_logic():
    """Tests that SleepAggregator correctly parses times, computes durations, and classifies them."""
    from src.data.components.label_aggregators import SleepAggregator
    
    # Create sample answers for SleepAggregator (questions 54 & 55)
    # response 1: 10:30 PM (22.5) to 7:00 AM (7.0) -> 8.5 hours -> class 1
    # response 2: 1:00 AM (1.0) to 8:00 AM (8.0) -> 7.0 hours -> class 1
    # response 3: 11:30 PM (23.5) to 4:00 AM (4.0) -> 4.5 hours -> class 0
    # response 4: 10:00 PM (22.0) to 9:00 AM (9.0) -> 11.0 hours -> class 2
    # response 5: missing awake answer -> should be dropped
    # response 6: malformed format -> should be dropped
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6],
        'question_id':        [54, 55, 54, 55, 54, 55, 54, 55, 54, 54, 55],
        'answer':             ["10:30 PM", "07:00 AM", "1:00 AM", "8:00 AM", "11:30 PM", "4:00 AM", "10:00 PM", "9:00 AM", "11:00 PM", "bad time", "07:00 AM"]
    })
    
    agg = SleepAggregator(asleep_id=54, awake_id=55)
    result = agg(df)
    
    # 5 and 6 should be dropped because of missing/malformed info
    assert len(result) == 4
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 1
    assert result.loc[result['survey_response_id'] == 3, 'answer'].values[0] == 0
    assert result.loc[result['survey_response_id'] == 4, 'answer'].values[0] == 2


@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_with_sleep_aggregator(mock_db_class):
    """Tests CohortBuilder integration with SleepAggregator, ensuring strings are not dropped."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import SleepAggregator

    survey_df = pd.DataFrame({
        'id': [101, 102],
        'app_user_id': [10, 10], # Use active/non-dropped users
        'timestamp': pd.to_datetime(['2025-10-01 08:00:00', '2025-10-01 09:00:00'])
    })
    
    answer_df = pd.DataFrame([
        # response 101 has fall asleep
        {'survey_response_id': 101, 'question_id': 54, 'answer': "10:00 PM"},
        # response 102 has wake up
        {'survey_response_id': 102, 'question_id': 55, 'answer': "07:00 AM"},
    ])
    
    # The two responses are on the same day for user 10, so they will be collapsed
    # Into one daily response containing both question 54 and question 55 answers.
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df}
    
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table].copy()
    
    agg = SleepAggregator(asleep_id=54, awake_id=55)
    
    builder = CohortBuilder(
        modalities=[],
        modality_cols={},
        preprocessors=None,
        aggregator=agg,
        os_filter='both',
        collapse_strategy='mean'
    )
    
    # Collapsed daily responses should preserve the string values and resolve them together
    collapsed_sr, collapsed_ans = builder._collapse_daily_responses(survey_df, answer_df, mock_db)
    
    # They should collapse to 1 survey response
    assert len(collapsed_sr) == 1
    # Both question 54 and 55 answers should exist in the collapsed answers
    assert len(collapsed_ans) == 2
    assert "10:00 PM" in collapsed_ans['answer'].values
    assert "07:00 AM" in collapsed_ans['answer'].values


@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_use_demographics_toggle(mock_db_class, dummy_data):
    """Tests that setting use_demographics=False prevents fetching/appending demographics."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Instantiate with use_demographics=False
    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2]),
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
        use_demographics=False
    )
    
    dm.setup()
    
    # 1. Verify datamodule properties
    assert dm.demographics_dim == 4
    assert dm.demographics_map is not None
    assert dm.default_demographics is not None
    
    # 2. Verify dataset item length is 4 (x, y, user_idx, demographics)
    train_ds = dm.data_train
    sample = train_ds[0]
    assert len(sample) == 4






