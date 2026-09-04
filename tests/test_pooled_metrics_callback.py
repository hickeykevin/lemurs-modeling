"""Integration tests for PooledMetricsCallback against a real Trainer.

Mirrors test_prediction_collector.py's pattern (real Trainer.fit()/test(),
not mocked), since PooledMetricsCallback composes PredictionCollectorCallback
and adds the on_test_end pooled-metrics/CI computation on top -- both need
covering end-to-end, not just the pure functions in
tests/test_pooled_metrics.py.
"""

import functools

import pandas as pd
import pytest
import torch
from lightning import Trainer

from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import OffsetSampler
from src.data.indexed_health_datamodule import IndexedHealthDataModule
from src.models.components.simple_lstm import SimpleLSTM
from src.models.health_module import HealthLitModule
from src.callbacks.evaluation_callbacks import ClassificationMetricsCallback
from src.callbacks.pooled_metrics_callback import PooledMetricsCallback



def _cohort(n_users=10, n_per_user=20):
    """Enough users/rows for a non-degenerate user-cluster bootstrap: half
    the users get a mostly-positive answer pattern, half mostly-negative."""
    rows = []
    step_rows = []
    for uid in range(n_users):
        is_positive_user = uid % 2 == 0
        for i in range(n_per_user):
            answer = 1 if (is_positive_user and i % 3 != 0) else 0
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * i),
                "answer": answer,
                "survey_response_id": uid * 1000 + i,
            })
        for i in range(n_per_user * 4):
            step_rows.append({
                "app_user_id": uid,
                "start_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=1.5 * i),
                "steps": 100,
            })
    master_df = pd.DataFrame(rows)
    step_df = pd.DataFrame(step_rows)
    demographics_df = pd.DataFrame(columns=["app_user_id", "gender", "age", "lgbt"])
    return {"step": step_df}, master_df, demographics_df


def _make_dm():
    modality_dfs, master_df, demographics_df = _cohort()
    return IndexedHealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        split_mode="longitudinal",
        train_val_test_split=(0.5, 0.2, 0.3),
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
def trained_run(tmp_path):
    dm = _make_dm()
    dm.setup()
    model = _make_model()
    classification_metrics = ClassificationMetricsCallback()
    pooled = PooledMetricsCallback(n_bootstraps=20)
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False, default_root_dir=str(tmp_path),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[classification_metrics, pooled],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return dm, pooled, tmp_path, classification_metrics


def test_pooled_predictions_match_data_test_count(trained_run):
    dm, pooled, _tmp_path, _cmc = trained_run
    assert len(pooled.pooled_predictions) == len(dm.data_test)


def test_pooled_metrics_include_all_metrics_and_cis(trained_run):
    _dm, pooled, _tmp_path, _cmc = trained_run
    for name in ("auroc", "auprc", "f1", "precision", "recall", "specificity", "sensitivity_at_specificity", "balanced_accuracy"):
        assert f"pooled/{name}" in pooled.pooled_metrics
        assert f"pooled/{name}_ci_low" in pooled.pooled_metrics
        assert f"pooled/{name}_ci_high" in pooled.pooled_metrics


def test_pooled_predictions_csv_is_written(trained_run):
    _dm, pooled, tmp_path, _cmc = trained_run
    out_path = tmp_path / "pooled_predictions.csv"
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert len(saved) == len(pooled.pooled_predictions)


def test_custom_output_filename_is_respected(tmp_path):
    dm = _make_dm()
    dm.setup()
    model = _make_model()
    pooled = PooledMetricsCallback(output_filename="custom_name.csv", n_bootstraps=10)
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False, default_root_dir=str(tmp_path),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[pooled],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    assert (tmp_path / "custom_name.csv").exists()
    assert not (tmp_path / "pooled_predictions.csv").exists()


def test_finds_and_reads_sibling_classification_metrics_callback_params(tmp_path):
    """PooledMetricsCallback must locate the ClassificationMetricsCallback
    attached to the SAME trainer and use its averaging/threshold params --
    not a second, independent set of defaults. Uses a non-default
    min_specificity so the assertion actually depends on the params having
    flowed through, not just "didn't crash"."""
    from src.eval_plans.pooled_metrics import _classification_metrics_params, _pooled_classification_metrics


    dm = _make_dm()
    dm.setup()
    model = _make_model()
    custom_cb = ClassificationMetricsCallback(min_specificity=0.5)
    pooled = PooledMetricsCallback(n_bootstraps=10)
    trainer = Trainer(
        max_epochs=1, accelerator="cpu", logger=False, default_root_dir=str(tmp_path),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[custom_cb, pooled],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    expected_params = _classification_metrics_params(custom_cb)
    assert expected_params["min_specificity"] == 0.5

    df = pooled.pooled_predictions
    y_true = df["y_true"].to_numpy()
    probs = df[["prob_class_0", "prob_class_1"]].to_numpy()
    recomputed = _pooled_classification_metrics(y_true, probs, expected_params)
    assert pooled.pooled_metrics["pooled/sensitivity_at_specificity"] == pytest.approx(
        recomputed["sensitivity_at_specificity"], abs=1e-6
    )
