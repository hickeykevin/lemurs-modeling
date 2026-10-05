import numpy as np
import pandas as pd

from src.data.components.samplers import MinuteGridSampler


def _run(records, lookback_hours=2.0, survey="2026-04-30 12:00:00", **kwargs):
    df = pd.DataFrame(
        {
            "app_user_id": [1] * len(records),
            "start_timestamp": pd.to_datetime([s for s, _, _ in records]),
            "end_timestamp": pd.to_datetime([e for _, e, _ in records]),
            "steps": [v for _, _, v in records],
        }
    )
    sampler = MinuteGridSampler(lookback_hours=lookback_hours, **kwargs)
    return sampler(pd.Timestamp(survey), 1, {"step": df}, {"step": "steps"}, ["step"])


def test_shape_and_empty_window():
    out = _run([])
    assert out.shape == (120, 2)
    assert out.sum() == 0


def test_long_record_spread_uniformly():
    # 600 steps over 10:30-11:30 (60 min) -> 10 steps/min on minutes 30..89.
    out = _run([("2026-04-30 10:30", "2026-04-30 11:30", 600)])
    values, observed = out[:, 0], out[:, 1]
    np.testing.assert_allclose(values[30:90], 10.0, rtol=1e-6)
    assert values[:30].sum() == 0 and values[90:].sum() == 0
    assert observed[30:90].all() and observed[:30].sum() == 0 and observed[90:].sum() == 0


def test_partial_minutes_and_total_preserved():
    # 90 steps over 10:00:30-10:02:00 (1.5 min): 30 in minute 0, 60 in minute 1.
    out = _run([("2026-04-30 10:00:30", "2026-04-30 10:02:00", 90)])
    np.testing.assert_allclose(out[:3, 0], [30.0, 60.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(out[:3, 1], [1.0, 1.0, 0.0])


def test_sub_minute_record_lands_in_one_minute():
    out = _run([("2026-04-30 10:05:10", "2026-04-30 10:05:40", 25)])
    assert out[5, 0] == 25.0
    assert out[:, 0].sum() == 25.0
    assert out[:, 1].sum() == 1.0


def test_point_record():
    out = _run([("2026-04-30 10:07:00", "2026-04-30 10:07:00", 12)])
    assert out[7, 0] == 12.0 and out[7, 1] == 1.0


def test_portion_outside_window_is_dropped():
    # 240 steps over 09:00-11:00; only 10:00-11:00 is inside -> 120 kept.
    out = _run([("2026-04-30 09:00", "2026-04-30 11:00", 240)])
    np.testing.assert_allclose(out[:, 0].sum(), 120.0, rtol=1e-6)
    np.testing.assert_allclose(out[:60, 0], 2.0, rtol=1e-6)


def test_max_record_minutes_masks_long_records():
    records = [
        ("2026-04-30 10:00", "2026-04-30 11:30", 900),  # 90 min -> dropped
        ("2026-04-30 11:40", "2026-04-30 11:45", 50),  # 5 min -> kept
    ]
    out = _run(records, max_record_minutes=60)
    assert out[:, 0].sum() == 50.0
    assert out[:, 1].sum() == 5.0


def test_survey_floored_to_align_freq():
    # Survey at 12:47 with 1h alignment -> window ends at 12:00.
    out = _run([("2026-04-30 12:10", "2026-04-30 12:20", 100)], survey="2026-04-30 12:47", align_freq="1h")
    assert out[:, 0].sum() == 0
