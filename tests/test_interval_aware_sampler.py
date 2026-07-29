import pytest
import pandas as pd
import numpy as np

from src.data.components.samplers import IntervalAwareSampler


ANCHOR = pd.Timestamp("2026-04-30 12:00:00")
USER = 1


def _sampler(**kw):
    defaults = dict(
        bin_edges_hours=[0, 3, 6, 12],
        emit_mask=True,
        normalize_by_duration=False,
        include_time_features=False,
    )
    defaults.update(kw)
    return IntervalAwareSampler(**defaults)


def _run(sampler, df):
    return sampler(
        survey_timestamp=ANCHOR,
        app_user_id=USER,
        modality_dfs={"step": df},
        modality_cols={"step": "steps"},
        modalities=["step"],
    )


def _df(rows):
    return pd.DataFrame(
        {
            "app_user_id": [USER] * len(rows),
            "start_timestamp": [pd.Timestamp(r[0]) for r in rows],
            "end_timestamp": [pd.Timestamp(r[1]) for r in rows],
            "steps": [r[2] for r in rows],
        }
    )


def test_bins_are_chronological_oldest_first():
    """Bin edges are hours *back* from the anchor, but output runs forward in time."""
    # Edges [0,3,6,12] -> bins [t-12,t-6), [t-6,t-3), [t-3,t)
    # 100 steps entirely within the most recent 3h must land in the LAST bin.
    df = _df([("2026-04-30 10:00:00", "2026-04-30 11:00:00", 100)])
    out = _run(_sampler(), df)

    assert out.shape == (3, 2)  # 3 bins, [value, mask]
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 100.0])


def test_value_splits_proportionally_across_bins():
    """A record straddling a boundary splits by overlap, not by start_timestamp."""
    # Bins are [t-12,t-6)=00:00-06:00, [t-6,t-3)=06:00-09:00, [t-3,t)=09:00-12:00.
    # A 600-step record spanning 07:00-13:00 is 6h long:
    #   overlap with 06:00-09:00 = 2h -> 2/6 of 600 = 200
    #   overlap with 09:00-12:00 = 3h -> 3/6 of 600 = 300
    # The final hour (12:00-13:00) is past the anchor and is dropped.
    df = _df([("2026-04-30 07:00:00", "2026-04-30 13:00:00", 600)])
    out = _run(_sampler(), df)

    np.testing.assert_allclose(out[:, 0], [0.0, 200.0, 300.0])


def test_out_of_window_mass_is_dropped_not_redistributed():
    """Portions of a record outside the lookback must not be pulled inward."""
    # 6h record, only its final hour (11:00-12:00) is inside the window at all.
    df = _df([("2026-04-30 06:00:00", "2026-04-30 12:00:00", 600)])
    out = _run(_sampler(bin_edges_hours=[0, 1]), df)

    # One bin [t-1h, t): 1/6 of the record's duration -> 100 steps.
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[:, 0], [100.0])


def test_start_bin_assignment_would_differ():
    """Guards the actual behavioural change vs. floor-and-groupby samplers."""
    # Record starts in the oldest bin but spans mostly into later ones.
    df = _df([("2026-04-30 01:00:00", "2026-04-30 11:00:00", 1000)])
    out = _run(_sampler(), df)

    # Start-bin assignment would put all 1000 in bin 0. Interval-aware spreads it.
    assert out[0, 0] < 1000.0
    assert out[1, 0] > 0.0 and out[2, 0] > 0.0
    np.testing.assert_allclose(out[:, 0].sum(), 1000.0)


def test_mask_separates_missing_from_sedentary():
    """A zero-valued observed bin and an unobserved bin must not look alike."""
    # A genuine zero-step record covering the middle bin only.
    df = _df([("2026-04-30 07:00:00", "2026-04-30 09:00:00", 0)])
    out = _run(_sampler(), df)

    values, mask = out[:, 0], out[:, 1]
    np.testing.assert_allclose(values, [0.0, 0.0, 0.0])
    # Middle bin [t-6,t-3) = 06:00-09:00 was covered; the others were not.
    np.testing.assert_allclose(mask, [0.0, 1.0, 0.0])


def test_mask_all_zero_when_user_has_no_data():
    df = _df([("2026-04-30 10:00:00", "2026-04-30 11:00:00", 500)])
    df["app_user_id"] = 999  # different user

    out = _run(_sampler(), df)
    np.testing.assert_allclose(out[:, 0], np.zeros(3))
    np.testing.assert_allclose(out[:, 1], np.zeros(3))


def test_normalize_by_duration_emits_comparable_rates():
    """Unequal-width bins share a feature column, so values must be rates."""
    # 300 steps in the 3h bin and 300 steps in the 6h bin are different rates.
    df = _df(
        [
            ("2026-04-30 09:30:00", "2026-04-30 10:30:00", 300),  # in [t-3,t)
            ("2026-04-30 03:00:00", "2026-04-30 04:00:00", 300),  # in [t-12,t-6)
        ]
    )
    out = _run(_sampler(normalize_by_duration=True), df)

    # Bin widths are 6h, 3h, 3h -> rates 50/h and 100/h.
    np.testing.assert_allclose(out[0, 0], 50.0)
    np.testing.assert_allclose(out[2, 0], 100.0)


def test_point_records_assigned_whole_to_containing_bin():
    """Zero-duration records have no interval to split."""
    df = _df([("2026-04-30 10:00:00", "2026-04-30 10:00:00", 250)])
    out = _run(_sampler(), df)

    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 250.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 0.0, 1.0])


def test_backwards_intervals_are_skipped():
    df = _df([("2026-04-30 11:00:00", "2026-04-30 09:00:00", 500)])
    out = _run(_sampler(), df)
    np.testing.assert_allclose(out[:, 0], np.zeros(3))


def test_multiple_modalities_interleave_value_and_mask():
    df = _df([("2026-04-30 10:00:00", "2026-04-30 11:00:00", 100)])
    cal = df.rename(columns={"steps": "calories"})
    cal["calories"] = 40

    out = _sampler()(
        survey_timestamp=ANCHOR,
        app_user_id=USER,
        modality_dfs={"step": df, "calorie": cal},
        modality_cols={"step": "steps", "calorie": "calories"},
        modalities=["calorie", "step"],
    )
    # [cal_value, cal_mask, step_value, step_mask]
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[2], [40.0, 1.0, 100.0, 1.0])


def test_time_features_appended_last():
    df = _df([("2026-04-30 10:00:00", "2026-04-30 11:00:00", 100)])
    out = _run(_sampler(include_time_features=True), df)
    # 1 modality * (value + mask) + 4 cyclic channels
    assert out.shape == (3, 6)
    assert np.all(np.abs(out[:, 2:]) <= 1.0)


@pytest.mark.parametrize("edges", [[5], [], [0, 3, 3], [0, 6, 3], [-1, 3]])
def test_invalid_bin_edges_rejected(edges):
    with pytest.raises(ValueError):
        IntervalAwareSampler(bin_edges_hours=edges)
