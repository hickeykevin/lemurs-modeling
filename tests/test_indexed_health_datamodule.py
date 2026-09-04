"""Tests for IndexedHealthDataModule: HealthDataModule with data_val/data_test
rebuilt to carry their sample index, for single-split (e.g. split_mode=
"longitudinal") runs that need PredictionCollectorCallback.

Uses prebuilt_cohort the same way test_walk_forward_datamodule.py does, since
the risk here is identical in shape: how setup() rebuilds data_val/data_test
with return_index=True after delegating to the base class, not the splitting
logic itself (already covered by test_data_components.py's CohortSplitter
tests and test_health_datamodule.py's base-class tests).
"""

import pandas as pd
import torch

from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import OffsetSampler
from src.data.indexed_health_datamodule import IndexedHealthDataModule


def _cohort(users=(1, 2), n_per_user=20):
    """Two users, hourly step data, surveys every 6h so a 6h-lookback sampler has coverage."""
    rows = []
    for uid in users:
        for i in range(n_per_user):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * i),
                "answer": i % 2,
                "survey_response_id": uid * 1000 + i,
            })
    master_df = pd.DataFrame(rows)

    step_rows = []
    for uid in users:
        for i in range(n_per_user * 4):
            step_rows.append({
                "app_user_id": uid,
                "start_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=1.5 * i),
                "steps": 100,
            })
    step_df = pd.DataFrame(step_rows)

    demographics_df = pd.DataFrame(columns=["app_user_id", "gender", "age", "lgbt"])
    return {"step": step_df}, master_df, demographics_df


def _make_dm(**overrides):
    modality_dfs, master_df, demographics_df = _cohort()
    kwargs = dict(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        split_mode="longitudinal",
        train_val_test_split=(0.5, 0.25, 0.25),
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    kwargs.update(overrides)
    return IndexedHealthDataModule(**kwargs)


def test_data_train_does_not_return_index():
    dm = _make_dm()
    dm.setup()
    assert dm.data_train.return_index is False


def test_data_val_and_data_test_return_index():
    dm = _make_dm()
    dm.setup()
    assert dm.data_val.return_index is True
    assert dm.data_test.return_index is True


def test_index_recovers_the_correct_source_row():
    dm = _make_dm()
    dm.setup()
    for i in range(len(dm.data_test)):
        sample = dm.data_test[i]
        idx_tensor = sample[-1]
        assert torch.is_tensor(idx_tensor) and idx_tensor.dtype == torch.long
        recovered = dm.data_test.data_links.iloc[int(idx_tensor)]
        assert recovered["app_user_id"] in (1, 2)


def test_setup_is_idempotent():
    dm = _make_dm()
    dm.setup()
    data_test_before = dm.data_test
    dm.setup()  # should no-op, not rebuild
    assert dm.data_test is data_test_before


def test_rebuilt_val_test_share_train_fitted_state_with_base_class():
    """The rebuilt data_val/data_test must reuse the SAME fitted scaler/
    demographics state the base class's own data_train build produced, not
    silently refit or diverge -- exact same regression this class's
    WalkForwardHealthDataModule counterpart guards against."""
    dm = _make_dm()
    dm.setup()
    assert dm.data_val.scaler is dm.data_train.scaler
    assert dm.data_test.scaler is dm.data_train.scaler
    assert dm.data_val.user_to_idx is dm.data_train.user_to_idx
    assert dm.data_test.user_to_idx is dm.data_train.user_to_idx


def test_user_split_mode_also_supported():
    """IndexedHealthDataModule is not longitudinal-specific -- any
    HealthDataModule split_mode should work unchanged. split_mode="user"
    needs enough distinct users for a 3-way split, so this uses a bigger
    synthetic cohort than the other tests' 2-user fixture."""
    modality_dfs, master_df, demographics_df = _cohort(users=tuple(range(1, 9)), n_per_user=10)
    dm = IndexedHealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        split_mode="user",
        train_val_test_split=(0.5, 0.25, 0.25),
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    dm.setup()
    assert dm.data_val.return_index is True
    assert dm.data_test.return_index is True
    assert len(dm.data_test) > 0 or len(dm.data_val) > 0
