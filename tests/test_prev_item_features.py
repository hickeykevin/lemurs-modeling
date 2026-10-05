"""Previous-survey item features: CohortBuilder._add_previous_items and config wiring."""

from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.label_aggregators import RuleBasedAggregator

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
THEME = RuleBasedAggregator(combination_logic="any", rules={"r": {"ids": [31, 32], "op": "sum", "threshold": 6}})


def _builder(lags=2):
    return CohortBuilder(
        modalities=[], modality_cols={}, preprocessors=None, aggregator=THEME,
        prev_item_aggregator=THEME, prev_item_lags=lags,
    )


def _stream():
    # User 1: responses 10, 11, 12 (t0 < t1 < t2, rows out of order); user 2: response 20.
    sr = pd.DataFrame(
        {
            "id": [12, 10, 20, 11],
            "app_user_id": [1, 1, 2, 1],
            "timestamp": pd.to_datetime(["2026-01-01 21:00", "2026-01-01 09:00", "2026-01-01 09:00", "2026-01-01 15:00"]),
        }
    )
    answers = pd.DataFrame(
        {
            "survey_response_id": [10, 10, 11, 11, 12, 12, 20, 20],
            "question_id": [31, 32] * 4,
            "answer": ["1", "2", "3", "4", "5", "5", "2", "yes"],
        }
    )
    return sr, answers


def _out(lags=2):
    sr, answers = _stream()
    master = pd.DataFrame({"survey_response_id": [10, 11, 12, 20], "app_user_id": [1, 1, 1, 2]})
    return _builder(lags)._add_previous_items(master, sr, answers).set_index("survey_response_id")


def test_lags_are_the_preceding_surveys_of_any_kind_in_time_order():
    out = _out()
    # Response 12 (21:00): lag 1 = response 11 (3, 4), lag 2 = response 10 (1, 2).
    assert (out.loc[12, "pi1_q31"], out.loc[12, "pi1_q32"]) == (3.0, 4.0)
    assert (out.loc[12, "pi2_q31"], out.loc[12, "pi2_q32"]) == (1.0, 2.0)
    assert out.loc[12, ["pi1_missing", "pi2_missing"]].tolist() == [0.0, 0.0]


def test_gap_hours_to_each_lagged_survey():
    out = _out()
    assert out.loc[12, "pi1_gap_h"] == 6.0 and out.loc[12, "pi2_gap_h"] == 12.0


def test_missing_history_is_flagged_and_zero_filled():
    out = _out()
    # First survey of a user: nothing before it.
    assert out.loc[10, ["pi1_missing", "pi2_missing"]].tolist() == [1.0, 1.0]
    assert out.loc[10, ["pi1_q31", "pi1_gap_h"]].tolist() == [0.0, 0.0]
    # Second survey has lag 1 but no lag 2.
    assert out.loc[11, ["pi1_missing", "pi2_missing"]].tolist() == [0.0, 1.0]
    # Histories never cross users.
    assert out.loc[20, ["pi1_missing", "pi2_missing"]].tolist() == [1.0, 1.0]


def test_target_surveys_own_items_are_never_features():
    out = _out(lags=1)
    # Response 11's answers (3, 4) appear only on the next survey (12), never on 11 itself.
    assert (out.loc[11, "pi1_q31"], out.loc[11, "pi1_q32"]) == (1.0, 2.0)


def test_columns_match_the_datamodule_naming():
    out = _out()
    assert [c for c in out.columns if c.startswith("pi")] == CohortBuilder.prev_item_columns([31, 32], 2)


def test_no_aggregator_is_a_noop():
    sr, answers = _stream()
    master = pd.DataFrame({"survey_response_id": [10], "app_user_id": [1]})
    b = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=THEME)
    assert b._add_previous_items(master, sr, answers).columns.tolist() == master.columns.tolist()


def test_config_group_selects_the_theme_questions():
    import hydra

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["data/prev_items=social_stress", "data/modalities=step"])
    assert sorted(hydra.utils.instantiate(cfg.data.prev_item_aggregator).get_question_ids()) == [31, 32, 33]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        assert compose(config_name="train").data.prev_item_aggregator is None


def _centred(mode):
    sr = pd.DataFrame({
        "id": [1, 2, 3, 4, 5], "app_user_id": [1] * 5,
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="12h"),
    })
    answers = pd.DataFrame({"survey_response_id": [1, 2, 3, 4, 5], "question_id": [31] * 5, "answer": ["1", "3", "3", "5", "1"]})
    b = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=THEME,
                      prev_item_aggregator=THEME, prev_item_lags=1, prev_item_centre=mode)
    m = pd.DataFrame({"survey_response_id": [1, 2, 3, 4, 5], "app_user_id": [1] * 5})
    return b._add_previous_items(m, sr, answers).set_index("survey_response_id")


def test_past_mean_centres_on_surveys_strictly_before_the_target():
    out = _centred("past_mean")
    # Target 4: lag-1 = survey 3 (3); usual = mean of surveys 1-3 = 7/3 -> deviation 3 - 7/3.
    assert abs(out.loc[4, "pi1_q31"] - (3 - 7 / 3)) < 1e-9
    # Target 2: lag-1 = survey 1 (1); usual over surveys before 2 = survey 1 -> 0.
    assert out.loc[2, "pi1_q31"] == 0.0
    # The target's own answer (survey 5 = 1) must not move its own centring.
    assert abs(out.loc[5, "pi1_q31"] - (5 - (1 + 3 + 3 + 5) / 4)) < 1e-9 + 0  # surveys 1-4


def test_past_z_is_zero_with_too_little_history_and_keeps_the_missing_flag():
    out = _centred("past_z")
    # Target 3 has two earlier surveys (1, 3): usual 2, sd sqrt(2); lag-1 = survey 2 (3).
    assert abs(out.loc[3, "pi1_q31"] - (3 - 2) / (2 ** 0.5)) < 1e-9
    assert out.loc[2, "pi1_q31"] == 0.0  # only one earlier survey
    assert out.loc[2, "pi1_missing"] == 0.0 and out.loc[1, "pi1_missing"] == 1.0


def test_invalid_centre_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=THEME, prev_item_centre="bogus")


def _sleep_out(centre="none"):
    # User 1: morning (reports sleep), afternoon (none), morning (reports sleep), target afternoon.
    sr = pd.DataFrame({
        "id": [1, 2, 3, 4], "app_user_id": [1] * 4,
        "timestamp": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 15:00", "2026-01-02 09:00", "2026-01-02 15:00"]),
    })
    answers = pd.DataFrame({
        "survey_response_id": [1, 1, 1, 2, 3, 3, 3, 4],
        "question_id": [31, 54, 55, 31, 31, 54, 55, 31],
        "answer": ["2", "10:00 PM", "6:00 AM", "3", "4", "11:00 PM", "5:00 AM", "1"],
    })
    b = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=THEME,
                      prev_item_aggregator=THEME, prev_item_lags=3, prev_item_centre=centre, prev_item_sleep=True)
    m = pd.DataFrame({"survey_response_id": [1, 2, 3, 4], "app_user_id": [1] * 4})
    return b._add_previous_items(m, sr, answers).set_index("survey_response_id")


def test_sleep_history_is_per_lag_and_missing_on_afternoon_surveys():
    out = _sleep_out()
    # Target 4: lag 1 = survey 3 (6h sleep), lag 2 = survey 2 (afternoon, no report), lag 3 = survey 1 (8h).
    assert out.loc[4, "pi1_sleep_hours"] == 6.0 and out.loc[4, "pi1_sleep_missing"] == 0.0
    assert out.loc[4, "pi2_sleep_hours"] == 0.0 and out.loc[4, "pi2_sleep_missing"] == 1.0
    assert out.loc[4, "pi3_sleep_hours"] == 8.0
    # The afternoon survey still has its theme items, so its item-missing flag stays 0.
    assert out.loc[4, "pi2_missing"] == 0.0 and out.loc[4, "pi2_q31"] == 3.0


def test_sleep_history_columns_match_the_datamodule_naming():
    out = _sleep_out()
    assert [c for c in out.columns if c.startswith("pi")] == CohortBuilder.prev_item_columns([31, 32], 3, True)


def test_sleep_history_with_past_z_uses_only_nights_before_the_target():
    out = _sleep_out("past_z")
    # Only one earlier night before target 3 -> too little history -> 0; target 4 has two nights (8h, 6h):
    assert out.loc[3, "pi1_sleep_hours"] == 0.0
    # usual 7h, sd sqrt(2); lag 1 = 6h -> (6-7)/sqrt(2)
    assert abs(out.loc[4, "pi1_sleep_hours"] - (-1 / 2 ** 0.5)) < 1e-9
