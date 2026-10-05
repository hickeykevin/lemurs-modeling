"""Log1pStandardScaler and MissingMaskSampler."""

import copy

import numpy as np
import pandas as pd

from src.data.components.samplers import MissingMaskSampler, RollingSampler
from src.data.components.scalers import Log1pStandardScaler


def test_log1p_standardizes_value_columns_and_passes_the_rest():
    X = np.array([[0.0, 0.0, 1.0], [np.e - 1, 9.0, 0.0], [np.e**2 - 1, 99.0, 1.0]])
    s = Log1pStandardScaler(scale_dim=2).fit(X)
    out = s.transform(X)
    z = np.log1p(X[:, 0])
    np.testing.assert_allclose(out[:, 0], (z - z.mean()) / z.std(), rtol=1e-6)
    np.testing.assert_allclose(out[:, :2].mean(0), 0.0, atol=1e-6)
    np.testing.assert_array_equal(out[:, 2], X[:, 2])  # mask column untouched


def _steps():
    return pd.DataFrame({
        "app_user_id": [1, 1, 1],
        # one record with 0 steps (data, no activity) and two with steps
        "start_timestamp": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 07:00", "2026-01-01 08:00"]),
        "end_timestamp": pd.to_datetime(["2026-01-01 02:00", "2026-01-01 08:00", "2026-01-01 09:00"]),
        "steps": [0.0, 100.0, 50.0],
    })


def test_mask_marks_bins_with_records_including_zero_valued_ones():
    inner = RollingSampler(lookback_hours=24, resample_freq="6h", include_time_features=True)
    s = MissingMaskSampler(copy.deepcopy(inner))
    dfs = {"step": _steps()}
    args = dict(survey_timestamp=pd.Timestamp("2026-01-02 00:30"), app_user_id=1,
                modality_dfs=dfs, modality_cols={"step": "steps"}, modalities=["step"])
    out = s(**args)
    ref = inner(**args)
    # bins: 00-06 (zero-valued record), 06-12 (150 steps), 12-18, 18-24 (no data)
    np.testing.assert_array_equal(out[:, 0], [0, 150, 0, 0])
    np.testing.assert_array_equal(out[:, 1], [1, 1, 0, 0])
    np.testing.assert_array_equal(out[:, 2:], ref[:, 1:])  # time features unchanged, last
    assert s.num_time_features == 4


def test_wrapper_forwards_sampler_settings():
    s = MissingMaskSampler(RollingSampler(lookback_hours=72, resample_freq="6h"))
    assert s.lookback_hours == 72 and s.resample_freq == "6h"
    s.include_time_features = False
    assert s.sampler.include_time_features is False and s.num_time_features == 0
    assert not hasattr(s, "set_labels")
    copy.deepcopy(s)
