"""Previous-survey label features: CohortBuilder._add_previous_labels and config wiring."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.label_aggregators import RuleBasedAggregator

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _ideation():
    return RuleBasedAggregator(combination_logic="any", rules={"r": {"ids": [2], "op": "ge", "val": 2.0}})


def _preparation():
    return RuleBasedAggregator(
        combination_logic="any",
        rules={"r1": {"ids": [7], "op": "ge", "val": 2.0}, "r2": {"ids": [8], "op": "any_gt", "val": 0}},
    )


def _builder():
    return CohortBuilder(
        modalities=[], modality_cols={}, preprocessors=None, aggregator=_ideation(),
        prev_label_aggregators={"suicide_ideation": _ideation(), "suicide_preparation": _preparation()},
    )


def _stream():
    # User 1: responses 10 (t0), 11 (t1), 12 (t2); user 2: response 20 (t0). Rows out of time order.
    sr = pd.DataFrame(
        {
            "id": [12, 10, 20, 11],
            "app_user_id": [1, 1, 2, 1],
            "timestamp": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-01", "2026-01-02"]),
        }
    )
    answers = pd.DataFrame(
        {
            "survey_response_id": [10, 10, 10, 11, 11, 11, 12, 12, 12, 20, 20, 20],
            "question_id": [2, 7, 8] * 4,
            #                 10: ideation only   11: preparation (yes)   12: nothing   20: both
            "answer": ["3", "1", "no", "1", "1", "yes", "1", "1", "no", "4", "5", "no"],
        }
    )
    return sr, answers


def test_previous_labels_come_from_the_users_preceding_survey():
    sr, answers = _stream()
    master = pd.DataFrame({"survey_response_id": [10, 11, 12, 20], "app_user_id": [1, 1, 1, 2]})
    out = _builder()._add_previous_labels(master, sr, answers).set_index("survey_response_id")

    # First survey of each user: no previous survey.
    assert out.loc[10, "prev_label_missing"] == 1.0 and out.loc[20, "prev_label_missing"] == 1.0
    assert out.loc[10, "prev_suicide_ideation"] == 0.0 and out.loc[20, "prev_suicide_preparation"] == 0.0
    # Response 11's previous survey is 10 (ideation positive, preparation negative).
    assert (out.loc[11, "prev_suicide_ideation"], out.loc[11, "prev_suicide_preparation"]) == (1.0, 0.0)
    assert out.loc[11, "prev_label_missing"] == 0.0
    # Response 12's previous survey is 11 (preparation positive via the yes/no item).
    assert (out.loc[12, "prev_suicide_ideation"], out.loc[12, "prev_suicide_preparation"]) == (0.0, 1.0)


def test_previous_survey_is_taken_from_the_full_stream_not_master_df():
    """Response 11 dropped from master_df (e.g. no sensor data) must still be
    response 12's previous survey."""
    sr, answers = _stream()
    master = pd.DataFrame({"survey_response_id": [10, 12], "app_user_id": [1, 1]})
    out = _builder()._add_previous_labels(master, sr, answers).set_index("survey_response_id")
    assert out.loc[12, "prev_suicide_preparation"] == 1.0  # from 11, not 10


def test_no_prev_aggregators_leaves_master_df_unchanged():
    sr, answers = _stream()
    master = pd.DataFrame({"survey_response_id": [10], "app_user_id": [1]})
    builder = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=_ideation())
    pd.testing.assert_frame_equal(builder._add_previous_labels(master, sr, answers), master)


@pytest.mark.parametrize("data_cfg", ["default", "user_cv", "walk_forward_cyclic"])
def test_prev_labels_config_group(data_cfg):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        off = compose(config_name="train", overrides=[f"data={data_cfg}"])
        on = compose(
            config_name="train",
            overrides=[f"data={data_cfg}", "data/prev_labels=suicide_ideation_preparation"],
        )
    assert off.data.prev_label_aggregators is None
    aggs = {k: hydra.utils.instantiate(v) for k, v in on.data.prev_label_aggregators.items()}
    assert set(aggs) == {"suicide_ideation", "suicide_preparation"}
    assert set(aggs["suicide_ideation"].get_question_ids()) == {2, 3, 5}
    assert set(aggs["suicide_preparation"].get_question_ids()) == {7, 8, 12, 13}


def test_health_dataset_appends_prev_label_columns():
    from src.data.components.health_dataset import HealthDataset
    from src.data.components.samplers import RollingSampler

    links = pd.DataFrame(
        {
            "app_user_id": [1, 1],
            "record_timestamp": pd.to_datetime(["2026-01-02 12:00", "2026-01-03 12:00"]),
            "answer": [0, 1],
            "prev_suicide_ideation": [0.0, 1.0],
            "prev_label_missing": [1.0, 0.0],
        }
    )
    steps = pd.DataFrame(
        {
            "app_user_id": [1],
            "start_timestamp": pd.to_datetime(["2026-01-02 06:00"]),
            "end_timestamp": pd.to_datetime(["2026-01-02 07:00"]),
            "steps": [100],
        }
    )
    ds = HealthDataset(
        links, {"step": steps}, {"step": "steps"}, RollingSampler(lookback_hours=24, resample_freq="6h"),
        None, prev_label_cols=["prev_suicide_ideation", "prev_label_missing"],
    )
    np.testing.assert_array_equal(ds[0]["demographics"].numpy(), [0.0, 1.0])
    np.testing.assert_array_equal(ds[1]["demographics"].numpy(), [1.0, 0.0])
