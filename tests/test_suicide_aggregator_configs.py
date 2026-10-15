"""The shipped suicide aggregator configs must not label a floor response positive.

Answers are stored 1-5 (see reports/database/updates/0014-fix-scale-answer-indexing.sql),
so a Likert rule of `ge 1` is true for every response. suicide_ideation.yaml and
suicide_preparation.yaml shipped with exactly that, labelling every survey positive.
"""

from pathlib import Path

import hydra
import pandas as pd
import pytest
from omegaconf import OmegaConf

AGGREGATOR_DIR = Path(__file__).resolve().parents[1] / "configs" / "data" / "aggregator"
SUICIDE_CONFIGS = ["suicide_risk", "suicide_ideation", "suicide_preparation"]
LIKERT_IDS = [2, 3, 5, 7]
YES_NO_IDS = [8, 12, 13]


def _aggregate(config: str, likert: str, yes_no: str) -> int:
    agg = hydra.utils.instantiate(OmegaConf.load(AGGREGATOR_DIR / f"{config}.yaml"))
    df = pd.DataFrame(
        {
            "survey_response_id": [1] * (len(LIKERT_IDS) + len(YES_NO_IDS)),
            "question_id": LIKERT_IDS + YES_NO_IDS,
            "answer": [likert] * len(LIKERT_IDS) + [yes_no] * len(YES_NO_IDS),
        }
    )
    # The pipeline coerces yes/no to 1/0 before aggregating (CohortBuilder).
    df["answer"] = df["answer"].map({"yes": 1.0, "no": 0.0}).fillna(pd.to_numeric(df["answer"], errors="coerce"))
    out = agg(df)
    return int(out.loc[out.survey_response_id == 1, "answer"].iloc[0])


@pytest.mark.parametrize("config", SUICIDE_CONFIGS)
def test_floor_response_is_negative(config):
    assert _aggregate(config, likert="1", yes_no="no") == 0


@pytest.mark.parametrize("config", SUICIDE_CONFIGS)
def test_likert_two_is_positive(config):
    assert _aggregate(config, likert="2", yes_no="no") == 1


@pytest.mark.parametrize("config", ["suicide_risk", "suicide_preparation"])
def test_yes_answer_is_positive(config):
    assert _aggregate(config, likert="1", yes_no="yes") == 1
