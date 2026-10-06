"""TabPFN as a FLAML learner: registration, FLAML plumbing, config; real model when licensed."""

import os

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.components import flaml_custom
from src.models.components.flaml_custom import TabPFNModel


def _data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return X, y


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(TabPFNModel, "_make_classifier", lambda self: LogisticRegression())


def test_wrapper_fits_ignores_weights_and_predicts(stand_in):
    X, y = _data()
    m = TabPFNModel(n_jobs=4).fit(X, y, sample_weight=np.ones(len(y)))
    p = m.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1) and set(m.classes_) == {0, 1}
    assert m.get_params()["model_version"] == "v3.5"
    assert m.set_params(model_version="v3.5-fast").model_version == "v3.5-fast"


def test_flaml_runs_the_tabpfn_learner(stand_in):
    from flaml import AutoML

    X, y = _data()
    automl = AutoML()
    flaml_custom.register_custom_learners(automl)
    automl.fit(X, y, task="classification", estimator_list=["tabpfn"], time_budget=10, metric="roc_auc",
               sample_weight=np.where(y == 1, 2.0, 1.0), verbose=0)
    assert automl.best_estimator == "tabpfn"
    assert automl.predict_proba(X).shape == (len(y), 2)


def test_config_selects_the_tabpfn_learner():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from src.utils.resolvers import register_resolvers

    register_resolvers()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=flaml_tabpfn"])
    assert list(cfg.model.automl_config.estimator_list) == ["tabpfn"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("TABPFN_TOKEN"), reason="needs the TabPFN licence token (TABPFN_TOKEN)")
def test_real_tabpfn_3_5_learns_a_simple_signal():
    pytest.importorskip("tabpfn")
    from sklearn.metrics import roc_auc_score

    X, y = _data(n=400)
    Xte, yte = _data(n=200, seed=1)
    m = TabPFNModel(model_version="v3.5").fit(X, y)
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.8
