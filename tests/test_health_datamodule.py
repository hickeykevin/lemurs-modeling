import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock, patch
from src.data.health_datamodule import HealthDataModule
from src.data.components.label_aggregators import MeanAggregator, MaxAggregator

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

def test_mean_aggregator():
    """Tests that MeanAggregator averages and binarizes correctly."""
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2],
        'answer': [1, 3, 0, 2]
    })
    
    # No binarization
    agg = MeanAggregator(threshold=None)
    result = agg(df)
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 2.0
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 1.0
    
    # With binarization (>= 2.0)
    agg_bin = MeanAggregator(threshold=2.0)
    result_bin = agg_bin(df)
    assert result_bin.loc[result_bin['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result_bin.loc[result_bin['survey_response_id'] == 2, 'answer'].values[0] == 0

def test_max_aggregator():
    """Tests that MaxAggregator correctly triggers if ANY answer meets threshold."""
    df = pd.DataFrame({
        'survey_response_id': [1, 1, 2, 2],
        'answer': [0, 1, 0, 0]
    })
    
    agg = MaxAggregator(threshold=1.0)
    result = agg(df)
    assert result.loc[result['survey_response_id'] == 1, 'answer'].values[0] == 1
    assert result.loc[result['survey_response_id'] == 2, 'answer'].values[0] == 0

@patch('src.data.health_datamodule.DatabaseService')
def test_datamodule_user_split(mock_db_class, dummy_data):
    """Tests that user split results in disjoint users."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Set up DM with user split (50/50 for simplicity in test)
    agg = MeanAggregator(threshold=1.0)
    dm = HealthDataModule(
        aggregator=agg,
        question_ids=[2, 4],
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

@patch('src.data.health_datamodule.DatabaseService')
def test_datamodule_longitudinal_split(mock_db_class, dummy_data):
    """Tests that longitudinal split respects chronology."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    dm = HealthDataModule(
        aggregator=MeanAggregator(threshold=None),
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

@patch('src.data.health_datamodule.DatabaseService')
def test_datamodule_multi_question_aggregation(mock_db_class, dummy_data):
    """Tests that labels are correctly aggregated from multiple question IDs."""
    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]
    
    # Mean of [2, 2] = 2.0. Threshold 1.5 -> Label 1
    dm = HealthDataModule(
        aggregator=MeanAggregator(threshold=1.5),
        question_ids=[2, 4]
    )
    
    dm.setup()
    
    # response_id 101 belongs to user 1
    row_101 = dm.data_train.data_links[dm.data_train.data_links['survey_response_id'] == 101]
    if row_101.empty: # might be in val/test
        row_101 = dm.data_val.data_links[dm.data_val.data_links['survey_response_id'] == 101]
    if row_101.empty:
        row_101 = dm.data_test.data_links[dm.data_test.data_links['survey_response_id'] == 101]
        
    assert row_101.iloc[0]['answer'] == 1
