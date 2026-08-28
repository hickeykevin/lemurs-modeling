"""Integration tests for PredictionCollectorCallback against a real Trainer.

Runs a genuine Trainer.fit()/test() over WalkForwardHealthDataModule rather
than mocking Lightning's hooks, since the actual risk here was end-to-end:
HealthDataset appends idx after demographics, and HealthLitModule.model_step
originally had no notion of it -- only a real fit/test call caught that (a
length-4 return_index=True batch was destructured as [x, y, _, demographics],
feeding the idx tensor into the network and crashing with a shape mismatch).
See test_health_module.py for isolated unit tests of the model_step fix.
"""

import functools

import pandas as pd
import pytest
import torch
from lightning import Trainer

from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import OffsetSampler
from src.data.walk_forward_health_datamodule import WalkForwardHealthDataModule
from src.models.components.simple_lstm import SimpleLSTM
from src.models.health_module import HealthLitModule
from src.utils.prediction_collector import PredictionCollectorCallback


def _cohort(users=(1, 2), n_per_user=20):
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


def _make_dm(current_fold=0):
    modality_dfs, master_df, demographics_df = _cohort()
    return WalkForwardHealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        fold_sizing="pct",
        burn_in_pct=0.3,
        step_pct=0.2,
        val_pct=0.1,
        current_fold=current_fold,
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )


def _make_model():
    net = SimpleLSTM(input_size=3, hidden_size=8, num_layers=1, output_size=2, use_sequence_data=True)
    return HealthLitModule(net=net, optimizer=functools.partial(torch.optim.Adam, lr=1e-3))


@pytest.fixture
def trained_fold():
    """Runs one epoch of fit + test on fold 0, with a collector attached."""
    dm = _make_dm(current_fold=0)
    dm.setup()
    model = _make_model()
    collector = PredictionCollectorCallback(fold_index=0)
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[collector],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return dm, collector


def test_fit_and_test_do_not_crash_with_return_index(trained_fold):
    """The end-to-end regression: idx must not reach the network as demographics."""
    dm, collector = trained_fold
    assert len(collector.rows) > 0


def test_collected_test_rows_match_data_test_count(trained_fold):
    dm, collector = trained_fold
    df = collector.to_dataframe()
    test_rows = df[df["stage"] == "test"]
    assert len(test_rows) == len(dm.data_test)


def test_collected_rows_join_back_to_correct_user_and_timestamp(trained_fold):
    dm, collector = trained_fold
    df = collector.to_dataframe()
    test_rows = df[df["stage"] == "test"].sort_values("sample_idx").reset_index(drop=True)

    for i, row in test_rows.iterrows():
        source = dm.data_test.data_links.iloc[int(row["sample_idx"])]
        assert row["app_user_id"] == source["app_user_id"]
        assert row["record_timestamp"] == source["record_timestamp"]


def test_probabilities_sum_to_one(trained_fold):
    _dm, collector = trained_fold
    df = collector.to_dataframe()
    prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
    assert len(prob_cols) == 2
    totals = df[prob_cols].sum(axis=1)
    assert (totals.sub(1.0).abs() < 1e-5).all()


def test_fold_index_tag_is_carried_on_every_row(trained_fold):
    _dm, collector = trained_fold
    df = collector.to_dataframe()
    assert (df["fold_index"] == 0).all()


def test_raises_without_return_index():
    """PredictionCollectorCallback must refuse to silently do nothing on a plain (non-walk-forward) datamodule."""
    from src.data.health_datamodule import HealthDataModule
    from src.data.components.cohort_splitter import CohortSplitter  # noqa: F401 (documents relationship)

    modality_dfs, master_df, demographics_df = _cohort()
    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        split_mode="longitudinal",
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        train_val_test_split=(0.6, 0.2, 0.2),
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    dm.setup()
    model = _make_model()
    collector = PredictionCollectorCallback()
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[collector],
    )
    # The collector raises the moment it sees a batch from a dataset that
    # wasn't built with return_index=True -- which happens during fit()'s own
    # validation pass here, before test() is ever reached. Either call site
    # demonstrates the guard; fit() is what actually triggers first.
    with pytest.raises(RuntimeError, match="return_index=True"):
        trainer.fit(model, datamodule=dm)
