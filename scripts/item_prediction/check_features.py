"""Sanity-check the previous-item features on the real cohort (one theme).

Reports: which survey types (morning=0, afternoon=1) answer the theme's items, how many target
surveys have 0..K preceding surveys, the gap distribution, and that no feature equals the
target survey's own items.

Usage (compute node): PYTHONPATH=. .venv/bin/python scripts/item_prediction/check_features.py social_stress
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.components.cohort_builder import CohortBuilder
from src.utils.database_service import DatabaseService

AGG = Path(__file__).resolve().parents[2] / "configs" / "data" / "aggregator"
theme = sys.argv[1] if len(sys.argv) > 1 else "social_stress"
K = 5
agg = hydra.utils.instantiate(OmegaConf.load(AGG / f"{theme}.yaml"))
db = DatabaseService()
b = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=agg,
                  collapse_strategy="none", prev_item_aggregator=agg, prev_item_lags=K)
m = b._extract_and_aggregate_labels(db)
ids = sorted(agg.get_question_ids())

print(f"theme={theme} items={ids} target surveys={len(m)} users={m.app_user_id.nunique()}")
print("answered theme items by survey type (survey_id):")
print(m.groupby("survey_id").size().to_string())
hist = (K - m[[f"pi{k}_missing" for k in range(1, K + 1)]].sum(axis=1)).astype(int)
print("\nnumber of preceding surveys with the theme's items (0..K):")
print(hist.value_counts().sort_index().to_string())
print(f"\nfull {K}-survey history: {(hist == K).mean():.1%} of targets")
gaps = m[[f"pi{k}_gap_h" for k in range(1, K + 1)]].where(m[[f"pi{k}_missing" for k in range(1, K + 1)]].values == 0)
print("\ngap to each lag (hours), median / 90th pct:")
print(pd.DataFrame({"median": gaps.median(), "p90": gaps.quantile(0.9)}).round(1).to_string())
print("\nmean item score at lag 1 vs lag 5 (where present):")
for q in ids:
    print(f"q{q}: lag1 {m.loc[m.pi1_missing == 0, f'pi1_q{q}'].mean():.2f}  lag5 {m.loc[m.pi5_missing == 0, f'pi5_q{q}'].mean():.2f}")
# the aggregator on the target survey and the lag-1 row must refer to different surveys
srt = m.sort_values(["app_user_id", "record_timestamp"])
prev_ts = srt.groupby("app_user_id").record_timestamp.shift(1)
ok = (srt.pi1_gap_h.round(3) == ((srt.record_timestamp - prev_ts).dt.total_seconds() / 3600).round(3)) | srt.pi1_missing.eq(1)
print(f"\nlag-1 gap equals the gap to the user's previous target row where present: {ok.mean():.1%}")
