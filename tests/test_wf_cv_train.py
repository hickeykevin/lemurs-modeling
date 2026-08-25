"""Integration tests for wf_cv_train.py against the real database.

Mirrors test_train.py's pattern (real DB via cfg_wf_cv_train, not mocked --
this project has no offline/synthetic-DB test fixture, matching test_train.py's
existing convention). Requires LEMURS_POSTGRES_* env vars to be set (source
.env). See tests/test_walk_forward_datamodule.py and
tests/test_prediction_collector.py for the unit/integration-level tests that
don't need a database (synthetic prebuilt_cohort fixtures).
"""

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.wf_cv_train import wf_cv_train


def test_wf_cv_train_fast_dev_run(cfg_wf_cv_train: DictConfig) -> None:
    """Runs the full walk-forward pipeline (every fold) with tightly limited
    batches, against the real cohort.

    This is the test that originally caught two real bugs while building this
    pipeline: (1) HealthLitModule.model_step misreading an appended idx tensor
    as demographics (see test_health_module.py / test_prediction_collector.py
    for the isolated regression tests), and (2) a walk-forward fold whose
    val_dataloader ends up empty (fewer than one batch, since val_responses is
    a small fixed count per user rather than a fraction of the whole cohort)
    crashing a strict EarlyStopping instead of just skipping early stopping
    for that fold -- fixed via configs/callbacks/walk_forward.yaml's
    early_stopping.strict: False. Neither bug was visible from
    dm.setup()-level tests alone; both only surfaced running every fold
    through a real Trainer.fit()/test() against the real cohort's actual
    per-user response-count distribution.
    """
    HydraConfig().set_config(cfg_wf_cv_train)
    metrics, obj = wf_cv_train(cfg_wf_cv_train)

    assert "pooled/auroc" in metrics or "pooled/n_predictions" in metrics
    pooled_predictions = obj["pooled_predictions"]
    assert len(pooled_predictions) > 0
    assert set(pooled_predictions["stage"]) == {"test"}

    # Every fold's test predictions must be able to trace back to a real
    # (app_user_id, record_timestamp) -- the entire reason return_index
    # exists.
    assert pooled_predictions["app_user_id"].notna().all()
    assert pooled_predictions["record_timestamp"].notna().all()

    fold_breakdown = obj["fold_breakdown"]
    assert len(fold_breakdown) == pooled_predictions["fold_index"].nunique()


def test_wf_cv_train_pooled_predictions_have_no_duplicate_test_role(cfg_wf_cv_train: DictConfig) -> None:
    """A given (app_user_id, record_timestamp) should be a test-role prediction
    in at most one fold -- the pooled-metric honesty property
    CohortSplitter.split_walk_forward guarantees at the splitter level (see
    test_cohort_splitter_walk_forward_no_response_is_a_test_example_twice),
    checked here end-to-end through the real pipeline."""
    HydraConfig().set_config(cfg_wf_cv_train)
    _metrics, obj = wf_cv_train(cfg_wf_cv_train)

    pooled_predictions = obj["pooled_predictions"]
    key_cols = ["app_user_id", "record_timestamp"]
    dupes = pooled_predictions.duplicated(subset=key_cols, keep=False)
    assert not dupes.any(), pooled_predictions[dupes][key_cols + ["fold_index"]]
