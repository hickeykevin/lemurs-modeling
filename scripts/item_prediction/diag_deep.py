"""Why does the item branch fail on test folds when validation looks fine?

Per user_cv fold: train TwoBranchNet (items only) for 30 epochs outside Lightning and print val and
test AUROC at several epochs, with variants (full, gap column zeroed, no BatchNorm), next to a
logistic regression on the same static vector.

Usage (compute node): PYTHONPATH=. .venv/bin/python scripts/item_prediction/diag_deep.py
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.components.two_branch_net import TwoBranchNet
from src.utils.resolvers import register_resolvers

register_resolvers()
cfg_dir = str(Path(__file__).resolve().parents[2] / "configs")


def tensors(ds):
    items = [ds[i] for i in range(len(ds))]
    return (torch.stack([s["demographics"] for s in items]), torch.stack([s["targets"] for s in items]).numpy())


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def run_net(variant, tr, va, te, lags, step, seed=0, epochs=30):
    torch.manual_seed(seed)
    net = TwoBranchNet(input_size=1, use_sensor_branch=False, use_static_extras=False)
    net.init_demographics(tr[0].shape[1]); net.init_item_sequence(lags, step)
    if variant == "no_bn":
        net.item_norm = torch.nn.Identity()
    def prep(d):
        d = d.clone()
        if variant == "no_gap":
            d.view(d.shape[0], lags, step)[..., -1] = 0.0
        return d
    xtr, xva, xte = prep(tr[0]), prep(va[0]), prep(te[0])
    ytr = torch.tensor(tr[1]).long()
    w = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)]).float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    out = {}
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(net(None, xtr[idx]), ytr[idx]).backward(); opt.step()
        if ep in (0, 4, 9, 29):
            net.eval()
            with torch.no_grad():
                pv = torch.softmax(net(None, xva), -1)[:, 1].numpy(); pt = torch.softmax(net(None, xte), -1)[:, 1].numpy()
            out[ep + 1] = (auc(va[1], pv), auc(te[1], pt))
    return out


rows = []
for fold in range(5):
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            "eval_plan=user_cv", "data/sampler=sweep_configs/rolling_72h_6h_mask", "data/modalities=step_calorie_distance",
            "data/scaler=log1p_standard", "data/prev_items=negative_emotion_plus_suicide_risk", f"data.current_fold={fold}",
            "data.collapse_strategy=none", "data.use_sleep=false", "data.use_demographics=false",
            "data.use_survey_context=false", "data.require_sensor_data=true", "trainer=cpu"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    tr, va, te = tensors(dm.data_train), tensors(dm.data_val), tensors(dm.data_test)
    lags, step = dm.prev_item_layout
    sc = StandardScaler().fit(tr[0].numpy())
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(tr[0].numpy()), tr[1])
    lr_v, lr_t = auc(va[1], lr.predict_proba(sc.transform(va[0].numpy()))[:, 1]), auc(te[1], lr.predict_proba(sc.transform(te[0].numpy()))[:, 1])
    print(f"fold {fold}: train {len(tr[1])} (pos {tr[1].mean():.2f}), val {len(va[1])} (pos {va[1].mean():.2f}), "
          f"test {len(te[1])} (pos {te[1].mean():.2f}) | logreg val {lr_v:.2f} test {lr_t:.2f}", flush=True)
    for variant in ("full", "no_gap", "no_bn"):
        res = run_net(variant, tr, va, te, lags, step)
        print("   ", variant.ljust(7), " ".join(f"ep{e}: v{v:.2f}/t{t:.2f}" for e, (v, t) in res.items()), flush=True)
