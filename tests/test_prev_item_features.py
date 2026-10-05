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
