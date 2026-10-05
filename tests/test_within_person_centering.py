"""FLAMLHealthModule within_person_centering='past_mean'."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.health_module import FLAMLHealthModule


class _DS:
    def __init__(self, users, times, values):
        self._sequences = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
        self.data_links = pd.DataFrame({"app_user_id": users, "record_timestamp": pd.to_datetime(times)})

    def __len__(self):
        return len(self._sequences)


def _module(train, val, test):
    m = FLAMLHealthModule(automl_config={}, within_person_centering="past_mean")
    dm = SimpleNamespace(data_train=train, data_val=val, data_test=test, sampler=None, modalities=["step"])
    m._trainer = SimpleNamespace(datamodule=dm)
    m.print = lambda *a, **k: None
    return m


def test_deviation_from_mean_of_earlier_surveys_across_splits():
    # User 1 in time order: 2 (test, day1), 4 (train, day2), 9 (val, day3). User 2: 10 (train, day1).
    train = _DS([1, 2], ["2026-01-02", "2026-01-01"], [4, 10])
    val = _DS([1], ["2026-01-03"], [9])
    test = _DS([1], ["2026-01-01"], [2])
    m = _module(train, val, test)
    m._build_centered_features()
    assert m._centered["test"].ravel().tolist() == [0.0]          # first survey
    assert m._centered["train"].ravel().tolist() == [2.0, 0.0]    # 4 - mean(2); user 2 first
    assert m._centered["val"].ravel().tolist() == [6.0]           # 9 - mean(2, 4)


def test_constant_person_level_is_removed():
    # Two users with very different but constant levels -> all deviations zero.
    train = _DS([1, 1, 2, 2], ["2026-01-01", "2026-01-02"] * 2, [100, 100, 1, 1])
    m = _module(train, None, None)
    m._build_centered_features()
    np.testing.assert_array_equal(m._centered["train"], 0.0)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        FLAMLHealthModule(automl_config={}, within_person_centering="full_mean")


def test_past_z_divides_by_sd_of_earlier_rows():
    from src.models.health_module import past_centre

    users = np.array([1, 1, 1, 1])
    times = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]).to_numpy()
    X = np.array([[1.0], [3.0], [5.0], [2.0]])
    z = past_centre(X, users, times, "past_z").ravel()
    # rows 0,1: < 2 earlier rows -> 0; row 2: (5 - 2) / sd(1,3)=1 -> 3; row 3: (2 - 3) / sd(1,3,5)
    np.testing.assert_allclose(z, [0.0, 0.0, 3.0, -1.0 / np.std([1, 3, 5])])


def test_past_z_removes_person_scale():
    from src.models.health_module import past_centre

    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 1))
    X = np.r_[base, 100 * base]  # same pattern, person 2 varies 100x more
    users = np.r_[np.ones(50), 2 * np.ones(50)]
    times = np.r_[np.arange(50), np.arange(50)]
    z = past_centre(X, users, times, "past_z")
    np.testing.assert_allclose(z[:50], z[50:], atol=1e-9)
