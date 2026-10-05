"""Sleep timing features, within-person deviations and morning-to-afternoon carry."""

import numpy as np
import pandas as pd
import pytest

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.health_dataset import SLEEP_MODE_COLS, sleep_mode_cols
from src.data.components.label_aggregators import RuleBasedAggregator


def _builder(carry):
    agg = RuleBasedAggregator(combination_logic="any", rules={"r": {"ids": [2], "op": "ge", "val": 2.0}})
    return CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=agg,
                         use_sleep=True, sleep_carry_to_afternoon=carry)


# User 1: three mornings (days 1-3), afternoons on day 1 (after two mornings) and day 4 (no morning).
SR = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "app_user_id": [1, 1, 1, 1, 1, 1],
    "survey_id": [0, 0, 1, 0, 0, 1],
    "timestamp": pd.to_datetime([
        "2026-01-01 08:00", "2026-01-01 10:00", "2026-01-01 15:00",
        "2026-01-02 09:00", "2026-01-03 09:00", "2026-01-04 15:00",
    ]),
})
ASLEEP, AWAKE = 54, 55
NIGHTS = {1: ("10:00 PM", "06:00 AM"), 2: ("11:00 PM", "07:00 AM"), 4: ("12:00 AM", "08:00 AM"), 5: ("02:00 AM", "10:00 AM")}
ANSWERS = pd.DataFrame(
    [(rid, ASLEEP, a) for rid, (a, _) in NIGHTS.items()] + [(rid, AWAKE, w) for rid, (_, w) in NIGHTS.items()],
    columns=["survey_response_id", "question_id", "answer"],
)


def _attach(carry):
    b = _builder(carry)
    return b._attach_sleep(b._compute_sleep_features(ANSWERS), SR).set_index("survey_response_id")


def test_timing_columns():
    f = _builder(False)._compute_sleep_features(ANSWERS).set_index("survey_response_id")
    assert f.loc[1, "sleep_hours"] == 8.0
    assert f.loc[1, "sleep_bedtime"] == 10.0  # 22:00 = 10 h after noon
    assert f.loc[5, "sleep_bedtime"] == 14.0  # 02:00 = 14 h after noon, no wrap
    assert f.loc[1, "sleep_waketime"] == 6.0
    assert f.loc[1, "sleep_midpoint"] == 14.0  # 22:00 + 4 h = 02:00


def test_deviations_use_only_earlier_nights():
    out = _attach(False)
    # Nights 1, 2 have < 2 earlier nights -> 0. Night 4: bedtime 12 vs earlier (10, 11): (12 - 10.5) / 0.5 = 3.
    assert out.loc[1, "sleep_bedtime_dev"] == 0.0 and out.loc[2, "sleep_bedtime_dev"] == 0.0
    assert out.loc[4, "sleep_bedtime_dev"] == pytest.approx(3.0)
    # Night 5: 14 vs (10, 11, 12): mean 11, sd sqrt(2/3)
    assert out.loc[5, "sleep_bedtime_dev"] == pytest.approx(3.0 / np.sqrt(2 / 3))
    assert (out.loc[[1, 2, 4, 5], "sleep_missing"] == 0).all()


def test_afternoon_takes_latest_same_day_morning_only():
    out = _attach(True)
    # Afternoon 3 (day 1) carries morning 2 (the later of 1 and 2), deviation included.
    assert out.loc[3, "sleep_hours"] == 8.0 and out.loc[3, "sleep_bedtime"] == 11.0
    assert out.loc[3, "sleep_bedtime_dev"] == out.loc[2, "sleep_bedtime_dev"]
    assert out.loc[3, "sleep_missing"] == 0.0
    # Afternoon 6 (day 4) has no same-day morning: not carried from day 3.
    assert 6 not in out.index


def test_no_carry_leaves_afternoons_out():
    out = _attach(False)
    assert 3 not in out.index and 6 not in out.index


def test_sleep_mode_cols():
    assert sleep_mode_cols("timing_deviation") == SLEEP_MODE_COLS["timing_deviation"]
    assert "sleep_missing" in sleep_mode_cols("timing_deviation")
    assert not any(c in sleep_mode_cols("timing_deviation") for c in ("sleep_hours_scaled", "sleep_bedtime"))
    assert sleep_mode_cols(None) == [] and sleep_mode_cols("hours_only") == ["sleep_hours_scaled"]
    assert sleep_mode_cols("something_else") == SLEEP_MODE_COLS["both"]  # previous fallback
