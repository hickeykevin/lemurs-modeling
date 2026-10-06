"""TiRexFeaturizer: output layout, DualScaler handling, summary-stat parity, caching."""

import numpy as np
import pytest
import torch

from src.data.components.tabular_features import SummaryStatsFeaturizer
from src.data.components.tirex_features import TiRexFeaturizer
from src.models.components import tirex_net
from tests.test_tirex_net import FakeTiRex

N, T, M = 5, 12, 2


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    tirex_net._EMBED_CACHE.clear()
    def fake_ensure(self, device):
        if self.model is None:
            self.model = FakeTiRex()
    monkeypatch.setattr(tirex_net.TiRexEncoderNet, "_ensure_model", fake_ensure)


def _x(dual=True, n_time=4):
    rng = np.random.default_rng(0)
    cols = [rng.random((N, T, M)) * 5]
    if dual:
        cols.append(rng.standard_normal((N, T, M)))
    cols.append(rng.standard_normal((N, T, n_time)))
    return np.concatenate(cols, axis=-1).astype(np.float32)


def test_layout_is_embedding_then_summary_stats_on_all_sensor_columns():
    f = TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=M)
    x = _x()
    out = f.transform(x)
    stats = SummaryStatsFeaturizer(exclude_last_n_cols=4).transform(x)
    assert out.shape == (N, 512 * M + stats.shape[1])
    np.testing.assert_allclose(out[:, 512 * M :], stats, rtol=1e-6)


def test_only_the_global_block_is_encoded():
    f = TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=M, append_summary_stats=False)
    x = _x()
    y = x.copy()
    y[:, :, M : 2 * M] = 0.0  # subject-scaled copy changes, global block does not
    np.testing.assert_allclose(f.transform(x), f.transform(y))


def test_without_dual_scaler_and_stats():
    f = TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=M, append_summary_stats=False, layer_pooling="all")
    assert f.transform(_x(dual=False)).shape == (N, 512 * 12 * M)


def test_single_sample_and_empty_split():
    f = TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=M)
    x = _x()
    assert f.transform(x[0]).shape == f.transform(x)[0].shape
    assert f.transform(x[:0]).shape == (0, f.transform(x).shape[1])


def test_repeat_transform_uses_the_cache():
    f = TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=M)
    x = _x()
    f.transform(x)
    calls = f._net.model.calls
    f.transform(x)
    assert f._net.model.calls == calls


def test_bad_modality_count_rejected():
    with pytest.raises(ValueError):
        TiRexFeaturizer(exclude_last_n_cols=4, num_modalities=3).transform(_x())


def test_config_instantiates():
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir

    from src.utils.resolvers import register_resolvers

    register_resolvers()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=flaml_rf", "model/featurizer=tirex"])
    assert isinstance(hydra.utils.instantiate(cfg.model.featurizer), TiRexFeaturizer)
