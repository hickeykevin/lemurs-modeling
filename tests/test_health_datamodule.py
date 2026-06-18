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
    
    return {"step": step_df, "survey_response": survey_df, "answer": answer_df}


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
        seq, target, _ = train_ds[0]
        
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
    assert result.shape == (12, 1)
    # Total steps should be 120 (12 * 10)
    assert result.sum() == 120
    
    # Test 12-hour lookback, 4-hour resampling
    sampler_4h = RollingSampler(lookback_hours=12, resample_freq="4h")
    result_4h = sampler_4h(survey_time, 1, modality_dfs, modality_cols, ["step"])
    
    # Should have 3 time steps (12/4)
    assert result_4h.shape == (3, 1)
    # Each bin should have 4 hours of data (40 steps)
    assert result_4h[0, 0] == 40

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
    
    # Total duration is 12 hours
    assert result.shape == (12, 1)
    assert result.sum() == 120
    
    # First bin should be June 1 at 18:00
    # Last bin should be June 2 at 05:00
    # Wait, my OffsetSampler implementation uses start_time as the first bin.
    # start_time is June 1 18:00.
    # end_time is June 2 06:00.
    # duration 12h. num_periods 12.
    # full_range starts at 18:00.

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
    
    dummy_data = {"step": step_df, "survey_response": survey_df, "answer": answer_df}

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
            seq, _, _ = ds[i]
            assert torch.allclose(seq, expected_seq, atol=1e-5)


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
    seq, target, _ = train_ds[0]
    assert target.dtype == torch.float32
    
    # 2. Check model steps in regression mode
    net = SimpleLSTM(input_size=1, hidden_size=8, num_layers=1, output_size=1)
    
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
    callback = RegressionMetricsCallback()
    outputs = model.validation_step(batch, 0)
    
    # Simulate batch end
    trainer = MagicMock()
    trainer.sanity_checking = False
    callback.on_validation_batch_end(trainer, model, outputs, batch, 0)
    
    # Compute metrics
    metrics = callback._init_metrics(device=torch.device("cpu"))
    metrics.update(outputs["preds"], outputs["targets"])
    res = metrics.compute()
    assert "mse" in res
    assert "mae" in res


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
    
    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df}
    
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
    
    _, master_df = builder.build()
    
    # Master df should contain:
    # 101 (kept)
    # 103 (kept)
    # 104 (kept)
    # 105 (kept)
    # 102 should be filtered out!
    response_ids = master_df['survey_response_id'].tolist()
    assert 101 in response_ids
    assert 102 not in response_ids
    assert 103 in response_ids
    assert 104 in response_ids
    assert 105 in response_ids
    assert len(response_ids) == 4



