import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from flaml import AutoML
from src.models.components.flaml_custom import SVMPipeline, SVMPipelineEstimator
from src.models.health_module import FLAMLHealthModule


@pytest.fixture(autouse=True)
def clean_hydra():
    """Ensure GlobalHydra is cleaned before and after tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


FLAML_MODEL_CONFIGS = [
    ("flaml_xgboost", ["xgboost"]),
    ("flaml_rf", ["rf"]),
    ("flaml_lr", ["lrl1", "lrl2"]),
    ("flaml_svm", ["svm_pipeline"]),
]


@pytest.mark.parametrize("config_name,expected_estimators", FLAML_MODEL_CONFIGS)
def test_flaml_model_configs_compose(config_name, expected_estimators):
    """Test that all FLAML model configs compose cleanly in Hydra and declare the correct estimator_list."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"model={config_name}", "paths.output_dir=/tmp"],
        )
        assert isinstance(cfg, DictConfig)
        assert cfg.model.task == "classification"
        assert list(cfg.model.automl_config.estimator_list) == expected_estimators

        # Test Hydra instantiation
        module = instantiate(cfg.model)
        assert isinstance(module, FLAMLHealthModule)


def test_svm_pipeline_direct():
    """Test SVMPipeline directly with scaling and sample weights."""
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, 30)
    weights = np.ones(30)

    pipe = SVMPipeline(C=0.5, kernel="rbf", gamma="scale")
    pipe.fit(X, y, sample_weight=weights)

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)
    assert preds.shape == (30,)
    assert probs.shape == (30, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("estimator_name", ["rf", "lrl2", "svm_pipeline"])
def test_flaml_estimators_fit_and_predict(estimator_name):
    """Test that AutoML successfully fits each individual estimator and yields calibrated predictions."""
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 2, 40)

    automl = AutoML()
    automl.add_learner("svm_pipeline", SVMPipelineEstimator)

    automl.fit(
        X,
        y,
        task="classification",
        estimator_list=[estimator_name],
        time_budget=2,
        metric="accuracy",
        verbose=0,
    )

    preds = automl.predict(X)
    probs = automl.predict_proba(X)

    assert preds.shape == (40,)
    assert probs.shape == (40, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_flaml_feature_extraction_without_demographics():
    """Verify that batches without demographics produce [B, T*F] shaped features."""
    import torch
    module = FLAMLHealthModule(automl_config={"time_budget": 1, "estimator_list": ["rf"]})
    B, T, F = 8, 10, 6
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    user_idx = torch.arange(B)

    batch = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
    }
    features, targets = module._extract_features_and_targets(batch, stage="train")
    assert features.shape == (B, T * F)
    assert torch.equal(targets, y)


def test_flaml_feature_extraction_with_demographics():
    """Verify that batches with demographics correctly concatenate into [B, (T*F) + D]."""
    import torch
    module = FLAMLHealthModule(automl_config={"time_budget": 1, "estimator_list": ["rf"]})
    B, T, F, D = 8, 10, 6, 11
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    user_idx = torch.arange(B)
    demographics = torch.randn(B, D)

    batch = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
        "demographics": demographics,
    }
    features, targets = module._extract_features_and_targets(batch, stage="train")
    assert features.shape == (B, (T * F) + D)
    assert torch.equal(targets, y)
    np.testing.assert_allclose(features[:, T * F:], demographics.numpy(), rtol=1e-5)


def test_flaml_feature_extraction_with_return_index():
    """Verify that sample_idx in dict batch does not interfere with feature extraction."""
    import torch
    module = FLAMLHealthModule(automl_config={"time_budget": 1, "estimator_list": ["rf"]})
    B, T, F, D = 8, 10, 6, 11
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    user_idx = torch.arange(B)
    demographics = torch.randn(B, D)
    row_idx = torch.arange(100, 100 + B, dtype=torch.long)

    # With demographics and sample_idx
    batch_with_demo = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
        "demographics": demographics,
        "sample_idx": row_idx,
    }
    features, targets = module._extract_features_and_targets(batch_with_demo, stage="val")
    assert features.shape == (B, (T * F) + D)
    assert torch.equal(targets, y)
    np.testing.assert_allclose(features[:, T * F:], demographics.numpy(), rtol=1e-5)

    # Without demographics and with sample_idx
    batch_no_demo = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
        "sample_idx": row_idx,
    }
    features_no_demo, targets_no_demo = module._extract_features_and_targets(batch_no_demo, stage="val")
    assert features_no_demo.shape == (B, T * F)
    assert torch.equal(targets_no_demo, y)


def test_flaml_fit_and_val_with_demographics():
    """Verify end-to-end setup (training) and validation with demographic features."""
    import torch
    module = FLAMLHealthModule(
        automl_config={"time_budget": 2, "estimator_list": ["rf"], "verbose": 0},
        task="classification",
    )
    B, T, F, D = 16, 5, 2, 4
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    user_idx = torch.arange(B)
    demo = torch.randn(B, D)

    class DummyDataset:
        return_index = False

    class DummyDataModule:
        def __init__(self):
            self.data_train = DummyDataset()
            self.data_val = DummyDataset()

        def train_dataloader(self):
            return [{
                "features": x,
                "targets": y,
                "user_indices": user_idx,
                "demographics": demo,
            }]

        def val_dataloader(self):
            return [{
                "features": x,
                "targets": y,
                "user_indices": user_idx,
                "demographics": demo,
            }]

    class DummyTrainer:
        def __init__(self):
            self.datamodule = DummyDataModule()
            self.should_stop = False
            self.is_global_zero = True
            self.progress_bar_callback = None

        def datamodule(self):
            return self.datamodule

    module.trainer = DummyTrainer()
    module.setup("fit")

    assert hasattr(module.automl, "best_estimator")
    val_out = module.validation_step({
        "features": x,
        "targets": y,
        "user_indices": user_idx,
        "demographics": demo,
    }, batch_idx=0)
    assert "preds" in val_out
    assert "logits" in val_out
    assert val_out["preds"].shape == (B,)


def test_flaml_feature_extraction_dict_batches():
    """Verify that modern dict batches are handled cleanly without positional unpacking."""
    import torch
    module = FLAMLHealthModule(automl_config={"time_budget": 1, "estimator_list": ["rf"]})
    B, T, F, D = 8, 10, 6, 11
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    user_idx = torch.arange(B)
    demographics = torch.randn(B, D)
    sample_idx = torch.arange(B, dtype=torch.long)

    # 1. Dict without demographics
    batch_no_demo = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
    }
    feats, targets = module._extract_features_and_targets(batch_no_demo, stage="train")
    assert feats.shape == (B, T * F)
    assert torch.equal(targets, y)

    # 2. Dict with demographics and sample_idx
    batch_with_demo = {
        "features": x,
        "targets": y,
        "user_indices": user_idx,
        "demographics": demographics,
        "sample_idx": sample_idx,
    }
    feats, targets = module._extract_features_and_targets(batch_with_demo, stage="val")
    assert feats.shape == (B, (T * F) + D)
    assert torch.equal(targets, y)
    np.testing.assert_allclose(feats[:, T * F:], demographics.numpy(), rtol=1e-5)


def test_flaml_feature_concatenation_exact_numeric_ordering():
    """Verify that columns 0..T*F-1 strictly contain flattened sequence data and columns

    T*F..(T*F)+D-1 strictly contain demographic data with zero column transposition.
    """
    import torch
    module = FLAMLHealthModule(automl_config={"time_budget": 1, "estimator_list": ["rf"]})
    B, T, F, D = 4, 3, 2, 5

    # Deterministic sequence values: each entry is unique
    x = torch.arange(B * T * F, dtype=torch.float32).reshape(B, T, F)
    # Deterministic demographic values: offset by 1000
    demo = torch.arange(B * D, dtype=torch.float32).reshape(B, D) + 1000.0
    y = torch.tensor([0, 1, 0, 1])

    batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": demo,
    }

    features, targets = module._extract_features_and_targets(batch, stage="train")
    expected_seq_flat = x.reshape(B, T * F).numpy()
    expected_demo = demo.numpy()

    # Verify overall shape
    assert features.shape == (B, (T * F) + D)
    assert features.shape == (4, 11)

    # Check exact numerical alignment of sequence slice
    np.testing.assert_array_equal(features[:, : T * F], expected_seq_flat)

    # Check exact numerical alignment of demographic slice
    np.testing.assert_array_equal(features[:, T * F :], expected_demo)


def test_flaml_xgboost_estimator_with_demographics():
    """Verify that the flaml_xgboost estimator trains on (T*F)+D features and outputs valid probabilities."""
    import torch
    B, T, F, D = 20, 4, 3, 6
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    demo = torch.randn(B, D)

    module = FLAMLHealthModule(
        automl_config={"time_budget": 3, "estimator_list": ["xgboost"], "verbose": 0},
        task="classification",
    )

    batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": demo,
    }

    class DummyDataModule:
        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    class DummyTrainer:
        datamodule = DummyDataModule()
        should_stop = False
        is_global_zero = True
        progress_bar_callback = None

    module.trainer = DummyTrainer()
    module.setup("fit")

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.best_estimator == "xgboost"

    # Verify that the underlying XGBoost model saw (T * F) + D = 18 features
    xgb_estimator = module.automl.model.estimator
    assert xgb_estimator.n_features_in_ == (T * F) + D
    assert xgb_estimator.n_features_in_ == 18

    # Verify validation predictions
    val_out = module.validation_step(batch, batch_idx=0)
    assert val_out["preds"].shape == (B,)
    assert val_out["logits"].shape == (B, 2)
    probs = val_out["logits"].softmax(dim=-1).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_flaml_end_to_end_walk_forward_pipeline_with_demographics(tmp_path):
    """Integration test: WalkForwardHealthDataModule -> dict batches -> FLAMLHealthModule.

    Verifies that real demographic processing produces 11 demographic dimensions,
    FLAML trains on all (T*F)+11 features, and PredictionCollectorCallback succeeds.
    """
    import functools
    import pandas as pd
    import torch
    from lightning import Trainer
    from src.callbacks.prediction_collector import PredictionCollectorCallback
    from src.data.components.label_aggregators import MeanAggregator
    from src.data.components.samplers import OffsetSampler
    from src.data.walk_forward_health_datamodule import WalkForwardHealthDataModule

    users = (1, 2, 3, 4)
    n_per_user = 20

    # 1. Build synthetic cohort with sensor modalities and demographic survey records
    master_rows = []
    for uid in users:
        for i in range(n_per_user):
            master_rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * i),
                "answer": i % 2,
                "survey_response_id": uid * 1000 + i,
            })
    master_df = pd.DataFrame(master_rows)

    step_rows = []
    calorie_rows = []
    distance_rows = []
    for uid in users:
        for i in range(n_per_user * 4):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=1.5 * i)
            step_rows.append({"app_user_id": uid, "start_timestamp": ts, "steps": 100})
            calorie_rows.append({"app_user_id": uid, "start_timestamp": ts, "calories": 50})
            distance_rows.append({"app_user_id": uid, "start_timestamp": ts, "distance": 0.5})

    modality_dfs = {
        "step": pd.DataFrame(step_rows),
        "calorie": pd.DataFrame(calorie_rows),
        "distance": pd.DataFrame(distance_rows),
    }

    demographics_df = pd.DataFrame([
        {"app_user_id": 1, "gender": "male", "age": 25, "lgbt": "no"},
        {"app_user_id": 2, "gender": "female", "age": 30, "lgbt": "yes"},
        {"app_user_id": 3, "gender": "male", "age": 35, "lgbt": "no"},
        {"app_user_id": 4, "gender": "female", "age": 40, "lgbt": "no"},
    ])

    # 2. Setup WalkForwardHealthDataModule with 3 modalities and demographics enabled
    dm = WalkForwardHealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0, resample_freq="1h"),
        modalities=["step", "calorie", "distance"],
        fold_sizing="pct",
        burn_in_pct=0.3,
        step_pct=0.2,
        val_pct=0.1,
        current_fold=0,
        use_demographics=True,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    dm.setup()

    # Verify datamodule demographics dimensions: 1 age + 3 gender + 3 lgbt + 4 source channels = 11
    assert dm.demographics_dim == 11

    # Verify batch shape emitted by train dataloader
    train_batch = next(iter(dm.train_dataloader()))
    assert isinstance(train_batch, dict)
    assert "features" in train_batch
    assert "targets" in train_batch
    assert "demographics" in train_batch
    assert "user_indices" in train_batch

    B, T, F = train_batch["features"].shape
    D = train_batch["demographics"].shape[1]
    assert F == 5  # 3 modalities (step, calorie, distance) + 2 time features (sin/cos hour)
    assert D == 11  # 11 demographic dimensions

    # 3. Setup FLAMLHealthModule with Random Forest and run fit + test with PredictionCollectorCallback
    module = FLAMLHealthModule(
        automl_config={"time_budget": 3, "estimator_list": ["rf"], "verbose": 0},
        task="classification",
    )
    collector = PredictionCollectorCallback(fold_index=0)

    trainer = Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[collector],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm)

    # 4. Verify fitted model ingested all (T * F) + D features
    rf_estimator = module.automl.model.estimator
    expected_total_features = (T * F) + D
    assert rf_estimator.n_features_in_ == expected_total_features
    assert rf_estimator.n_features_in_ == (T * 5) + 11

    # 5. Verify prediction collector gathered valid rows joined to source
    collected_df = collector.to_dataframe()
    assert len(collected_df) > 0
    test_rows = collected_df[collected_df["stage"] == "test"]
    assert len(test_rows) == len(dm.data_test)
    assert "sample_idx" in test_rows.columns
    assert "prob_class_0" in test_rows.columns
    assert "prob_class_1" in test_rows.columns


@pytest.mark.parametrize("estimator_name", ["xgboost", "rf", "lrl2", "svm_pipeline"])
def test_flaml_all_tabular_estimators_with_demographics(estimator_name):
    """Verify that all four FLAML tabular estimators (xgboost, rf, lrl2, svm) train on (T*F)+D
    features, record expected feature counts, and produce valid (B,) and (B, 2) shaped outputs."""
    import torch
    B, T, F, D = 24, 4, 3, 5
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    demo = torch.randn(B, D)

    module = FLAMLHealthModule(
        automl_config={"time_budget": 2, "estimator_list": [estimator_name], "verbose": 0},
        task="classification",
    )

    batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": demo,
    }

    class DummyDataModule:
        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    class DummyTrainer:
        datamodule = DummyDataModule()
        should_stop = False
        is_global_zero = True
        progress_bar_callback = None

    module.trainer = DummyTrainer()
    module.setup("fit")

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.best_estimator == estimator_name

    # Check underlying feature count
    model_est = module.automl.model.estimator
    if hasattr(model_est, "n_features_in_"):
        assert model_est.n_features_in_ == (T * F) + D
    elif hasattr(model_est, "named_steps"):
        assert model_est.named_steps["scaler"].n_features_in_ == (T * F) + D

    # Validate predictions and logits shapes
    val_out = module.validation_step(batch, batch_idx=0)
    assert val_out["preds"].shape == (B,)
    assert val_out["logits"].shape == (B, 2)
    probs = val_out["logits"].softmax(dim=-1).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Validate predict_step shape
    preds = module.predict_step(batch, batch_idx=0)
    assert preds.shape == (B,)


def test_flaml_single_sample_inference_shape():
    """Verify that single-sample inference (B=1) preserves 2D matrix shape and returns (1,) predictions."""
    import torch
    B_train, T, F, D = 20, 5, 2, 4
    x_train = torch.randn(B_train, T, F)
    y_train = torch.randint(0, 2, (B_train,))
    demo_train = torch.randn(B_train, D)

    module = FLAMLHealthModule(
        automl_config={"time_budget": 2, "estimator_list": ["rf"], "verbose": 0},
        task="classification",
    )

    train_batch = {
        "features": x_train,
        "targets": y_train,
        "user_indices": torch.arange(B_train),
        "demographics": demo_train,
    }

    class DummyDataModule:
        def train_dataloader(self):
            return [train_batch]

        def val_dataloader(self):
            return [train_batch]

    class DummyTrainer:
        datamodule = DummyDataModule()
        should_stop = False
        is_global_zero = True
        progress_bar_callback = None

    module.trainer = DummyTrainer()
    module.setup("fit")

    # Evaluate on single sample B=1
    single_batch = {
        "features": torch.randn(1, T, F),
        "targets": torch.tensor([1]),
        "user_indices": torch.tensor([0]),
        "demographics": torch.randn(1, D),
    }

    val_out = module.validation_step(single_batch, batch_idx=0)
    assert val_out["preds"].shape == (1,)
    assert val_out["logits"].shape == (1, 2)

    pred_out = module.predict_step(single_batch, batch_idx=0)
    assert pred_out.shape == (1,)


def test_flaml_demographics_dimension_mismatch_raises():
    """Verify that passing an unexpected demographic feature dimension raises ValueError rather than silent corruption."""
    import torch
    B, T, F, D = 20, 4, 2, 4
    x = torch.randn(B, T, F)
    y = torch.randint(0, 2, (B,))
    demo = torch.randn(B, D)

    module = FLAMLHealthModule(
        automl_config={"time_budget": 2, "estimator_list": ["rf"], "verbose": 0},
        task="classification",
    )

    batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": demo,
    }

    class DummyDataModule:
        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    class DummyTrainer:
        datamodule = DummyDataModule()
        should_stop = False
        is_global_zero = True
        progress_bar_callback = None

    module.trainer = DummyTrainer()
    module.setup("fit")

    # Pass batch with mismatching demographic dimension D=10 instead of D=4
    bad_batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": torch.randn(B, 10),
    }

    with pytest.raises(ValueError, match=r"(features|dimensions|X has)"):
        module.validation_step(bad_batch, batch_idx=0)


def test_flaml_regression_with_demographics():
    """Verify FLAML regression task handles continuous targets and outputs (B,) shaped predictions."""
    import torch
    B, T, F, D = 25, 4, 3, 5
    x = torch.randn(B, T, F)
    # Continuous targets
    y = torch.randn(B)
    demo = torch.randn(B, D)

    module = FLAMLHealthModule(
        automl_config={"time_budget": 2, "estimator_list": ["rf"], "verbose": 0},
        task="regression",
    )

    batch = {
        "features": x,
        "targets": y,
        "user_indices": torch.arange(B),
        "demographics": demo,
    }

    class DummyDataModule:
        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    class DummyTrainer:
        datamodule = DummyDataModule()
        should_stop = False
        is_global_zero = True
        progress_bar_callback = None

    module.trainer = DummyTrainer()
    module.setup("fit")

    assert hasattr(module.automl, "best_estimator")
    assert module.automl.model.estimator.n_features_in_ == (T * F) + D

    val_out = module.validation_step(batch, batch_idx=0)
    assert val_out["preds"].shape == (B,)
    assert val_out["preds"].dtype == torch.float32

    pred_out = module.predict_step(batch, batch_idx=0)
    assert pred_out.shape == (B,)


