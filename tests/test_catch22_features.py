import numpy as np
import pytest

from src.data.components.catch22_features import Catch22Featurizer


def test_transform_shape_batched():
    x = np.random.randn(5, 10, 3).astype(np.float32)
    featurizer = Catch22Featurizer()
    out = featurizer.transform(x)
    assert out.shape == (5, 3 * len(featurizer.stats))
    assert out.dtype == np.float32


def test_transform_shape_single_sample():
    x = np.random.randn(10, 3).astype(np.float32)
    featurizer = Catch22Featurizer()
    out = featurizer.transform(x)
    assert out.shape == (3 * len(featurizer.stats),)


def test_no_crash_and_finite_output_at_short_lengths():
    # T=1 through T=8, the range this project's sweeps actually use.
    for t in range(1, 9):
        x = np.random.randn(2, t, 2).astype(np.float32)
        featurizer = Catch22Featurizer()
        out = featurizer.transform(x)
        assert out.shape == (2, 2 * len(featurizer.stats))
        assert np.all(np.isfinite(out))  # fill_value replaces any NaN/inf


def test_constant_window_is_finite_after_fill():
    # Most catch22 features are undefined on a constant series; fill_value
    # should replace all of them rather than propagating NaN.
    x = np.full((1, 8, 1), 5.0, dtype=np.float32)
    featurizer = Catch22Featurizer(fill_value=-1.0)
    out = featurizer.transform(x)
    assert np.all(np.isfinite(out))
    assert np.sum(out == -1.0) > 0  # at least some features hit the fill


def test_exclude_last_n_cols_drops_trailing_columns():
    x = np.random.randn(3, 8, 5).astype(np.float32)  # e.g. 1 modality + 4 time cols
    featurizer = Catch22Featurizer(exclude_last_n_cols=4)
    out = featurizer.transform(x)
    assert out.shape == (3, 1 * len(featurizer.stats))


def test_exclude_all_cols_raises():
    x = np.random.randn(2, 8, 4).astype(np.float32)
    featurizer = Catch22Featurizer(exclude_last_n_cols=4)
    with pytest.raises(ValueError):
        featurizer.transform(x)


def test_empty_time_dimension_returns_zeros():
    x = np.zeros((3, 0, 2), dtype=np.float32)
    featurizer = Catch22Featurizer()
    out = featurizer.transform(x)
    assert out.shape == (3, 2 * len(featurizer.stats))
    assert np.all(out == 0.0)


def test_feature_names_length_matches_output():
    featurizer = Catch22Featurizer()
    names = featurizer.feature_names(["step", "calorie"])
    x = np.random.randn(2, 8, 2).astype(np.float32)
    out = featurizer.transform(x)
    assert len(names) == out.shape[1]
