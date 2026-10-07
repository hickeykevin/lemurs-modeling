"""Check W&B grid sweeps against their full parameter grid.

For every combination a sweep config defines, reports whether a run finished with pooled
metrics, finished WITHOUT them (e.g. died before the last fold but was marked finished),
crashed/failed, is still running, or never ran. Writes the combinations that need a rerun to
``<out>/<sweep name>_reruns.txt`` as ``key=value`` lines (one run per line), so they can be
replayed through the sweep's wrapper script.

Usage:
    .venv/bin/python scripts/check_sweep_completeness.py [--ids logs/fm_sweep_ids.json] [--out logs/sweep_checks]
"""

import argparse
import collections
import itertools
import json
import netrc
from pathlib import Path

import requests
import yaml

ENTITY, PROJECT = "hickeykevin", "lemurs-modeling"
QUERY = """query S($e:String!,$p:String!,$n:String!){project(entityName:$e,name:$p){sweep(sweepName:$n){
  state runs(first:2000){edges{node{name displayName state summaryMetrics config}}}}}}"""


def headers():
    auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")
    return {"Authorization": f"Bearer {auth[2]}", "Content-Type": "application/json"}


def grid(cfg):
    params = cfg["parameters"]
    swept = {k: v["values"] for k, v in params.items() if "values" in v}
    fixed = {k: v["value"] for k, v in params.items() if "value" in v}
    keys = list(swept)
    return keys, [dict(zip(keys, combo)) for combo in itertools.product(*swept.values())], fixed


def unwrap(cfg, key):
    v = cfg.get(key)
    return v.get("value") if isinstance(v, dict) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="logs/fm_sweep_ids.json")
    ap.add_argument("--out", default="logs/sweep_checks")
    args = ap.parse_args()
    ids = json.load(open(args.ids))
    ids = ids.get("current", ids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = headers()

    print("| Sweep | sweep state | grid | finished + metrics | finished, no metrics | crashed/failed | running | never ran |")
    print("|---|---|---|---|---|---|---|---|")
    details = {}
    for name, sid in ids.items():
        cfg = yaml.safe_load(open(f"configs/hparams_search/{name}.yaml"))
        keys, combos, fixed = grid(cfg)
        r = requests.post("https://api.wandb.ai/graphql", headers=H,
                          json={"query": QUERY, "variables": {"e": ENTITY, "p": PROJECT, "n": sid}}, timeout=120).json()
        sw = r["data"]["project"]["sweep"]
        by_combo = collections.defaultdict(list)
        for e in sw["runs"]["edges"]:
            n = e["node"]
            c = json.loads(n["config"] or "{}")
            s = json.loads(n["summaryMetrics"] or "{}")
            key = tuple(str(unwrap(c, k)) for k in keys)
            by_combo[key].append((n["state"], s.get("pooled/auroc") is not None, n["name"], n["displayName"]))

        cats = collections.defaultdict(list)
        for combo in combos:
            runs = by_combo.get(tuple(str(combo[k]) for k in keys), [])
            if any(st == "finished" and ok for st, ok, *_ in runs):
                cats["ok"].append(combo)
            elif any(st == "running" for st, *_ in runs):
                cats["running"].append(combo)
            elif any(st == "finished" for st, *_ in runs):
                cats["no_metrics"].append((combo, [r for r in runs if r[0] == "finished"]))
            elif runs:
                cats["crashed"].append((combo, runs))
            else:
                cats["never"].append(combo)
        details[name] = cats
        print(f"| {name} ({sid}) | {sw['state']} | {len(combos)} | {len(cats['ok'])} | {len(cats['no_metrics'])} | "
              f"{len(cats['crashed'])} | {len(cats['running'])} | {len(cats['never'])} |")

        reruns = [c for c, _ in cats["no_metrics"]] + [c for c, _ in cats["crashed"]]
        if sw["state"] != "RUNNING":
            reruns += cats["never"]
        lines = [" ".join(f"{k}={v}" for k, v in {**c, **fixed}.items()) for c in reruns]
        (out / f"{name}_reruns.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    for name, cats in details.items():
        for label, key in (("finished without pooled metrics", "no_metrics"), ("crashed/failed", "crashed")):
            for combo, runs in cats[key]:
                print(f"{name} | {label}: {combo} | runs: {[(r[2], r[0]) for r in runs]}")
    print(f"\nrerun lists written to {out}/<sweep>_reruns.txt")


if __name__ == "__main__":
    main()
