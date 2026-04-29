import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.components.samplers import BlockSampler

def test_block_sampler_interval_overlap():
    survey_timestamp = pd.Timestamp("2026-04-30 12:00:00")
    user_id = 1
    
    # Lookback 1 day:
    # Target date: 2026-04-29
    # B1 (Sleep): 00:00 - 08:00
    # B2 (Morning): 08:00 - 12:00
    # B3 (Afternoon): 12:00 - 17:00
    # B4 (Evening): 17:00 - 24:00
    
    df_step = pd.DataFrame({
        'app_user_id': [user_id, user_id, user_id],
        'start_timestamp': [
            pd.Timestamp("2026-04-29 09:00:00"),  # Fully in Morning
            pd.Timestamp("2026-04-29 10:00:00"),  # Crosses Morning -> Afternoon
            pd.Timestamp("2026-04-29 22:00:00")   # Crosses Evening -> Midnight
        ],
        'end_timestamp': [
            pd.Timestamp("2026-04-29 11:00:00"),
            pd.Timestamp("2026-04-29 15:00:00"),
            pd.Timestamp("2026-04-30 01:00:00")
        ],
        'steps': [500, 1000, 300]
    })
    
    # Distribution Math:
    # Rec 1: 500 steps fully in B2 (Morning).
    # Rec 2: 1000 steps over 5 hours (300 mins).
    #   - B2 overlap: 10:00 to 12:00 (120 mins). 120/300 = 40% = 400 steps.
    #   - B3 overlap: 12:00 to 15:00 (180 mins). 180/300 = 60% = 600 steps.
    # Rec 3: 300 steps over 3 hours (180 mins).
    #   - B4 overlap: 22:00 to 24:00 (120 mins). 120/180 = 66.67% = 200 steps.
    #   - Beyond 24:00 cutoff (60 mins) is ignored.
    
    modality_dfs = {'step': df_step}
    modality_cols = {'step': 'steps'}
    modalities = ['step']
    
    sampler = BlockSampler(lookback_days=1)
    features = sampler(survey_timestamp, user_id, modality_dfs, modality_cols, modalities)
    
    # Expected Array:
    # B1: 0
    # B2: 500 + 400 = 900
    # B3: 600
    # B4: 292.6829
    
    expected = np.array([[0.0], [900.0], [600.0], [292.6829]], dtype=np.float32)
    np.testing.assert_array_almost_equal(features, expected, decimal=3)



def test_block_sampler_point_fallback():
    survey_timestamp = pd.Timestamp("2026-04-30 12:00:00")
    user_id = 1
    
    # Test point logic fallback when 'end_timestamp' is missing
    df_point = pd.DataFrame({
        'app_user_id': [user_id, user_id],
        'start_timestamp': [
            pd.Timestamp("2026-04-29 07:00:00"), # Sleep Block
            pd.Timestamp("2026-04-29 14:00:00")  # Afternoon Block
        ],
        'val': [50.0, 75.0]
    })
    
    modality_dfs = {'point_mod': df_point}
    modality_cols = {'point_mod': 'val'}
    modalities = ['point_mod']
    
    sampler = BlockSampler(lookback_days=1)
    features = sampler(survey_timestamp, user_id, modality_dfs, modality_cols, modalities)
    
    expected = np.array([[50.0], [0.0], [75.0], [0.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(features, expected)

def test_block_sampler_comprehensive_day():
    survey_timestamp = pd.Timestamp("2026-04-30 12:00:00")
    user_id = 1
    
    # Simulation setup (10 records):
    # 7 Short Records (5-10 mins):
    # 1. Sleep: 02:00-02:10 (150 steps)
    # 2. Sleep: 07:15-07:20 (200 steps)
    # 3. Morning: 08:30-08:40 (600 steps)
    # 4. Morning: 11:45-11:55 (400 steps)
    # 5. Afternoon: 13:15-13:20 (300 steps)
    # 6. Evening: 18:30-18:40 (800 steps)
    # 7. Evening: 21:00-21:10 (500 steps)
    
    # 3 Long Records (>3 hours crossing blocks):
    # 8. Morning -> Afternoon: 09:00 to 14:00 (5 hrs / 300 mins total). 3000 steps.
    #    - Overlap B2 (09:00-12:00): 180 mins. 180/300 * 3000 = 1800 steps.
    #    - Overlap B3 (12:00-14:00): 120 mins. 120/300 * 3000 = 1200 steps.
    # 9. Afternoon -> Evening: 15:00 to 19:00 (4 hrs / 240 mins total). 2000 steps.
    #    - Overlap B3 (15:00-17:00): 120 mins. 120/240 * 2000 = 1000 steps.
    #    - Overlap B4 (17:00-19:00): 120 mins. 120/240 * 2000 = 1000 steps.
    # 10. Evening -> Sleep: 22:00 to 02:00 (4 hrs / 240 mins total). 1000 steps.
    #    - Overlap B4 (22:00-24:00): 120 mins. 120/240 * 1000 = 500 steps.
    
    df_step = pd.DataFrame({
        'app_user_id': [user_id] * 10,
        'start_timestamp': [
            pd.Timestamp("2026-04-29 02:00:00"),
            pd.Timestamp("2026-04-29 07:15:00"),
            pd.Timestamp("2026-04-29 08:30:00"),
            pd.Timestamp("2026-04-29 11:45:00"),
            pd.Timestamp("2026-04-29 13:15:00"),
            pd.Timestamp("2026-04-29 18:30:00"),
            pd.Timestamp("2026-04-29 21:00:00"),
            pd.Timestamp("2026-04-29 09:00:00"),
            pd.Timestamp("2026-04-29 15:00:00"),
            pd.Timestamp("2026-04-29 22:00:00")
        ],
        'end_timestamp': [
            pd.Timestamp("2026-04-29 02:10:00"),
            pd.Timestamp("2026-04-29 07:20:00"),
            pd.Timestamp("2026-04-29 08:40:00"),
            pd.Timestamp("2026-04-29 11:55:00"),
            pd.Timestamp("2026-04-29 13:20:00"),
            pd.Timestamp("2026-04-29 18:40:00"),
            pd.Timestamp("2026-04-29 21:10:00"),
            pd.Timestamp("2026-04-29 14:00:00"),
            pd.Timestamp("2026-04-29 19:00:00"),
            pd.Timestamp("2026-04-30 02:00:00")
        ],
        'steps': [150, 200, 600, 400, 300, 800, 500, 3000, 2000, 1000]
    })
    
    modality_dfs = {'step': df_step}
    modality_cols = {'step': 'steps'}
    modalities = ['step']
    
    sampler = BlockSampler(lookback_days=1)
    features = sampler(survey_timestamp, user_id, modality_dfs, modality_cols, modalities)
    
    # Expected Array:
    # B1 (Sleep): 150 + 200 = 350
    # B2 (Morning): 600 + 400 + 1800 = 2800
    # B3 (Afternoon): 300 + 1200 + 1000 = 2500
    # B4 (Evening): 800 + 500 + 1000 + 952.3809 = 3252.3809
    
    expected = np.array([[350.0], [2800.0], [2500.0], [3252.3809]], dtype=np.float32)
    np.testing.assert_array_almost_equal(features, expected, decimal=3)



