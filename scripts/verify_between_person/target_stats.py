"""How much does each candidate EMA target vary within people?

For every binary aggregator config (and its regression twin, where one exists),
labels are built exactly as CohortBuilder builds them, then restricted to the
surveys in the analysis cohort (the test surveys of the cyclical XGBoost run).

Reports, per target:
  * item answer range (sanity check of the 1-5 scale against the threshold)
  * base rate, participants whose label ever changes, onset events
  * ICC of the continuous score: share of variance that is between people
    (high = mostly a trait, little for a within-person model to find)
  * persistence: P(pos | prev pos) and P(pos | prev neg)

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/target_stats.py
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.components.cohort_builder import CohortBuilder
from src.utils.database_service import DatabaseService

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "configs" / "data" / "aggregator"
COHORT = REPO / "logs" / "verify_between_person" / "runs" / "xgboost" / "pooled_predictions.csv"
TARGETS = [
    "suicide_risk", "suicide_ideation", "negative_emotion", "positive_emotion",
    "social_stress", "social_connection", "emotion_regulation", "minority_stress", "self_harm",
]


def labels_for(name: str, db: DatabaseService) -> pd.DataFrame:
    agg = hydra.utils.instantiate(OmegaConf.load(AGG / f"{name}.yaml"))
    builder = CohortBuilder(
        modalities=[], modality_cols={}, preprocessors=None, aggregator=agg, collapse_strategy="none"  # as the runs
    )
    df = builder._extract_and_aggregate_labels(db)
    return df[["survey_response_id", "app_user_id", "record_timestamp", "answer"]]


def item_range(name: str, db_answers: pd.DataFrame) -> str:
    agg = hydra.utils.instantiate(OmegaConf.load(AGG / f"{name}.yaml"))
    a = pd.to_numeric(db_answers[db_answers.question_id.isin(agg.get_question_ids())].answer, errors="coerce").dropna()
    return f"{a.min():.0f}-{a.max():.0f}" if len(a) else "yes/no"


def icc(score: pd.Series, users: pd.Series) -> float:
    """One-way random-effects ICC(1)."""
    d = pd.DataFrame({"s": score, "u": users}).dropna()
    groups = d.groupby("u").s
    k = groups.size().mean()
    msb = (groups.size() * (groups.mean() - d.s.mean()) ** 2).sum() / (groups.ngroups - 1)
    msw = ((d.s - groups.transform("mean")) ** 2).sum() / (len(d) - groups.ngroups)
    return (msb - msw) / (msb + (k - 1) * msw)


def main():
    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort.stage == "test"][["app_user_id", "record_timestamp"]].drop_duplicates()
    cohort["record_timestamp"] = pd.to_datetime(cohort.record_timestamp).dt.floor("s")  # CSV keeps ms only

    db = DatabaseService()
    assert db.connect()
    try:
        answers = db.extract_from_database("answer")
        rows = []
        for name in TARGETS:
            if not (AGG / f"{name}.yaml").exists():
                continue
            lab = labels_for(name, db)
            lab["record_timestamp"] = pd.to_datetime(lab.record_timestamp).dt.floor("s")
            lab = lab.merge(cohort, on=["app_user_id", "record_timestamp"]).sort_values(["app_user_id", "record_timestamp"])
            y = lab.answer.astype(float)
            prev = lab.groupby("app_user_id").answer.shift(1)
            varies = lab.groupby("app_user_id").answer.nunique() > 1
            score_icc = np.nan
            if (AGG / f"{name}_regression.yaml").exists():
                reg = labels_for(f"{name}_regression", db)
                reg["record_timestamp"] = pd.to_datetime(reg.record_timestamp).dt.floor("s")
                reg = reg.merge(cohort, on=["app_user_id", "record_timestamp"])
                score_icc = icc(reg.answer.astype(float), reg.app_user_id)
            rows.append(dict(
                target=name,
                items=item_range(name, answers),
                surveys=len(lab),
                users=lab.app_user_id.nunique(),
                base_rate=y.mean(),
                users_varying=f"{varies.sum()}/{len(varies)}",
                onsets=int(((prev == 0) & (y == 1)).sum()),
                icc_score=score_icc,
                icc_label=icc(y, lab.app_user_id),
                p_pos_prev_pos=y[prev == 1].mean(),
                p_pos_prev_neg=y[prev == 0].mean(),
            ))
            print(f"done {name}", flush=True)
    finally:
        db.disconnect()
    out = pd.DataFrame(rows).set_index("target")
    print()
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
