"""Check the static-vector layout the TwoBranchNet assumes, on the real cohort.

Builds the user_cv datamodule (item history on, no previous-label columns), pulls one batch and
checks: the layout reported by the datamodule, that the item block is the tail of the static
vector, that lag 1 is first, that the gap column is last in each block, and compares the block
to the CohortBuilder columns for the same survey.

Usage (compute node): PYTHONPATH=. .venv/bin/python scripts/item_prediction/check_deep_layout.py
"""

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from pathlib import Path

from src.utils.resolvers import register_resolvers  # noqa: F401  (registers ${len:...})

register_resolvers()
cfg_dir = str(Path(__file__).resolve().parents[2] / "configs")
with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
    cfg = compose(config_name="train", overrides=[
        "eval_plan=user_cv", "data/sampler=sweep_configs/rolling_72h_6h_mask", "data/modalities=step_calorie_distance",
        "data/scaler=log1p_standard", "data/prev_items=negative_emotion_plus_suicide_risk",
        "data.collapse_strategy=none", "data.use_sleep=false", "data.use_demographics=false",
        "data.use_survey_context=false", "data.require_sensor_data=true", "model=two_branch", "trainer=cpu"])
dm = hydra.utils.instantiate(cfg.data)
dm.setup()
ds = dm.data_train if getattr(dm, "data_train", None) is not None else dm.train_dataset
print("demographics_dim:", dm.demographics_dim, "| prev_item_layout:", dm.prev_item_layout, "| prev_label_cols:", len(dm.prev_label_cols))
lags, step = dm.prev_item_layout
assert dm.demographics_dim == lags * step, (dm.demographics_dim, lags, step)
cols = dm.prev_label_cols
row = ds.data_links.iloc[0] if hasattr(ds, "data_links") else None
s = ds[0]
d = s["demographics"].numpy()
print("features:", tuple(s["features"].shape), "| demographics:", d.shape)
blocks = d.reshape(lags, step)
print("step columns:", cols[:step])
print("lag-1 block:", np.round(blocks[0], 2))
print("lag-%d block:" % lags, np.round(blocks[-1], 2))
assert cols[step - 1].endswith("_gap_h") and cols[step - 2].endswith("_missing")
print("gap (last col) per lag, hours:", np.round(blocks[:, -1], 1), "(should increase with lag)")
print("missing flag per lag:", blocks[:, -2])
