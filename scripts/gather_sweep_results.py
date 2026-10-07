"""Gather, extract, and consolidate results from W&B sweeps (--set standard, the default):
- 9dvq8uiy: RQ1 FLAML Models (560 combinations)
- a6sl4omd: RQ1 DL Models (LSTM, Transformer) (280 combinations)
- zoznzd8y: RQ2 FLAML Features (SummaryStats vs Catch22) (160 combinations)

and the foundation-model sweeps with --set foundation (TabPFN / TiRex; see FOUNDATION_SWEEPS).

Saves a consolidated CSV and per-sweep CSVs to reports/sweep_results/ (standard) or
reports/sweep_results_foundation/ (foundation).
"""

import json
import netrc
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

ENTITY = "hickeykevin"
PROJECT = "lemurs-modeling"

SWEEPS = {
    "9dvq8uiy": {
        "experiment": "RQ1_flaml_models",
        "description": "RQ1 FLAML Models x Rolling Samplers x Modalities",
        "default_featurizer": "raw",
        "model_type": "FLAML",
    },
    "a6sl4omd": {
        "experiment": "RQ1_dl_models",
        "description": "RQ1 Deep Learning Models (LSTM, Transformer)",
        "default_featurizer": "none (raw sequences)",
        "model_type": "DeepLearning",
    },
    "zoznzd8y": {
        "experiment": "RQ2_flaml_features",
        "description": "RQ2 FLAML Features (SummaryStats vs Catch22)",
        "default_featurizer": "summary_stats",
        "model_type": "FLAML",
        "fixed_modalities": "step_distance",
    },
}

# Foundation-model sweeps (2026-10-07, data snapshot data/db_snapshots/2026-10-07). IDs from
# logs/fm_sweep_ids.json ("current"); configs in configs/hparams_search/RQ*_FM_*.yaml.
FOUNDATION_SWEEPS = {
    "yaz6r51w": {"experiment": "RQ1_FM_tabpfn", "description": "RQ1 TabPFN-3.5 frozen + fine-tuned",
                 "default_featurizer": "raw", "model_type": "FoundationTabular"},
    "gvbc2t3n": {"experiment": "RQ2_FM_tabpfn", "description": "RQ2 TabPFN-3.5 on summary_stats / catch22",
                 "default_featurizer": "summary_stats", "model_type": "FoundationTabular", "fixed_modalities": "step_distance"},
    "gm4eoxqq": {"experiment": "RQ2_FM_tirex_featurizer", "description": "RQ2 FLAML models on frozen TiRex embeddings",
                 "default_featurizer": "tirex", "model_type": "FLAML", "fixed_modalities": "step_distance"},
    "hyyfwc68": {"experiment": "RQ1_FM_tirex", "description": "RQ1 TiRex frozen + linear head",
                 "default_featurizer": "none (raw sequences)", "model_type": "FoundationSequential"},
    "4tn2se89": {"experiment": "RQ1_FM_tirex_ft", "description": "RQ1 TiRex fine-tuned (last 2 blocks) + linear head",
                 "default_featurizer": "none (raw sequences)", "model_type": "FoundationSequential"},
}

SWEEP_SETS = {
    "standard": (SWEEPS, "reports/sweep_results", "all_sweeps_results.csv"),
    "foundation": (FOUNDATION_SWEEPS, "reports/sweep_results_foundation", "all_foundation_sweeps_results.csv"),
}

GRAPHQL_QUERY = """
query SweepRuns($entity: String!, $project: String!, $name: String!) {
  project(entityName: $entity, name: $project) {
    sweep(sweepName: $name) {
      name
      state
      runs(first: 2000) {
        edges {
          node {
            name
            displayName
            state
            summaryMetrics
            config
            createdAt
            heartbeatAt
          }
        }
      }
    }
  }
}
"""


def get_auth_headers() -> Dict[str, str]:
    netrc_path = Path.home() / ".netrc"
    auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    if not auth:
        raise RuntimeError("No W&B credentials found in ~/.netrc")
    return {"Authorization": f"Bearer {auth[2]}", "Content-Type": "application/json"}


def unwrap_val(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def parse_sampler(sampler_str: Optional[str]):
    if not sampler_str:
        return None, None
    m = re.search(r"rolling_(\d+)h_(\d+)h", str(sampler_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def count_modalities(mod_str: Optional[str]) -> Optional[int]:
    if not mod_str:
        return None
    # Can be 'step', 'step_calorie', 'step_calorie_distance', etc.
    known = ["step", "calorie", "distance"]
    count = sum(1 for k in known if k in str(mod_str))
    return count if count > 0 else 1


def metric_columns(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The metric columns of one run, from its W&B summary.

    The first block keeps the original export's columns and order. Later columns are appended
    so earlier exports stay a prefix of later ones.
    """
    g = summary.get
    recall = g("pooled/recall", g("pooled/sensitivity", g("test/pooled_sensitivity")))
    specificity = g("pooled/specificity", g("test/pooled_specificity"))
    balanced_accuracy = g("pooled/balanced_accuracy", g("test/pooled_balanced_accuracy"))
    if balanced_accuracy is None and recall is not None and specificity is not None:
        balanced_accuracy = (float(recall) + float(specificity)) / 2.0
    return {
        "pooled_auroc": g("pooled/auroc", g("test/pooled_auroc")),
        "pooled_f1": g("pooled/f1", g("test/pooled_f1")),
        "pooled_precision": g("pooled/precision", g("test/pooled_precision")),
        "pooled_recall": recall,
        "pooled_specificity": specificity,
        # Logged as pooled/sensitivity_at_specificity (min specificity 0.9, see
        # configs/callbacks/classification_metrics.yaml); the older key never existed.
        "pooled_sens_at_spec_0.9": g("pooled/sensitivity_at_specificity", g("pooled/sensitivity_at_spec_0.9")),
        # No pooled within-person AUROC is logged to W&B (only the last fold's value, which
        # would be misleading here); these stay empty. See reports/sweep_identity_decomposition.csv.
        "within_person_auroc_mean": g("within_person/auroc_mean", g("test/within_person_auroc_mean")),
        "within_person_auroc_median": g("within_person/auroc_median", g("test/within_person_auroc_median")),
        "fold_test_auroc_mean": g("fold/test/auroc_mean"),
        "fold_test_f1_mean": g("fold/test/f1_mean"),
        "fold_test_spec_mean": g("fold/test/specificity_mean"),
        "pooled_auroc_ci_low": g("pooled/auroc_ci_low"),
        "pooled_auroc_ci_high": g("pooled/auroc_ci_high"),
        # --- appended (2026-10-07) ---
        "pooled_balanced_accuracy": balanced_accuracy,
        "fold_test_balanced_accuracy_mean": g("fold/test/balanced_accuracy_mean"),
        "pooled_balanced_accuracy_ci_low": g("pooled/balanced_accuracy_ci_low"),
        "pooled_balanced_accuracy_ci_high": g("pooled/balanced_accuracy_ci_high"),
        "pooled_auprc": g("pooled/auprc"),
        "pooled_auprc_ci_low": g("pooled/auprc_ci_low"),
        "pooled_auprc_ci_high": g("pooled/auprc_ci_high"),
        "pooled_f1_ci_low": g("pooled/f1_ci_low"),
        "pooled_f1_ci_high": g("pooled/f1_ci_high"),
        "pooled_precision_ci_low": g("pooled/precision_ci_low"),
        "pooled_precision_ci_high": g("pooled/precision_ci_high"),
        "pooled_recall_ci_low": g("pooled/recall_ci_low"),
        "pooled_recall_ci_high": g("pooled/recall_ci_high"),
        "pooled_specificity_ci_low": g("pooled/specificity_ci_low"),
        "pooled_specificity_ci_high": g("pooled/specificity_ci_high"),
        "pooled_sens_at_spec_0.9_ci_low": g("pooled/sensitivity_at_specificity_ci_low"),
        "pooled_sens_at_spec_0.9_ci_high": g("pooled/sensitivity_at_specificity_ci_high"),
        "fold_test_precision_mean": g("fold/test/precision_mean"),
        "fold_test_recall_mean": g("fold/test/recall_mean"),
        "fold_test_sens_at_spec_0.9_mean": g("fold/test/sensitivity_at_specificity_mean"),
        "pooled_n_users": g("pooled/n_users"),
        "pooled_n_predictions": g("pooled/n_predictions"),
        "pooled_n_positive": g("pooled/n_positive"),
    }


METRIC_COLUMNS = list(metric_columns({}).keys())
STANDARD_EXTRA_COLUMNS = METRIC_COLUMNS[METRIC_COLUMNS.index("pooled_balanced_accuracy_ci_low"):]


def post(headers, query, variables):
    resp = requests.post("https://api.wandb.ai/graphql", headers=headers,
                         json={"query": query, "variables": variables}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_all_sweeps(sweeps: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    headers = get_auth_headers()
    all_records = []

    for sweep_id, meta in sweeps.items():
        print(f"\nFetching sweep '{sweep_id}' ({meta['experiment']})...")
        data = post(headers, GRAPHQL_QUERY, {"entity": ENTITY, "project": PROJECT, "name": sweep_id})
        sweep_data = data.get("data", {}).get("project", {}).get("sweep")
        if not sweep_data:
            print(f"Warning: Sweep '{sweep_id}' not found in W&B project {ENTITY}/{PROJECT}")
            continue

        edges = sweep_data.get("runs", {}).get("edges", [])
        print(f"  -> Sweep State: {sweep_data.get('state')} | Total runs fetched: {len(edges)}")

        for edge in edges:
            node = edge["node"]
            raw_summary = node.get("summaryMetrics") or "{}"
            raw_config = node.get("config") or "{}"
            summary = json.loads(raw_summary) if isinstance(raw_summary, str) else raw_summary
            config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config

            sampler_pkg = unwrap_val(config, "sampler_pkg")
            model_pkg = unwrap_val(config, "model_pkg")
            modalities_choice = unwrap_val(config, "modalities_choice")
            featurizer_pkg = unwrap_val(config, "featurizer_pkg", meta["default_featurizer"])
            if not modalities_choice:
                # Sweeps with a fixed modality set (the RQ2 featurizer sweeps) do not sweep it.
                modalities_choice = meta.get("fixed_modalities")
                if not modalities_choice:
                    mod_cfg = unwrap_val(config, "data/modalities") or unwrap_val(config, "data.modalities")
                    modalities_choice = str(mod_cfg) if mod_cfg else None

            lookback_h, resample_h = parse_sampler(sampler_pkg)
            all_records.append({
                "sweep_id": sweep_id,
                "experiment": meta["experiment"],
                "model_type": meta["model_type"],
                "run_id": node.get("name"),
                "run_name": node.get("displayName"),
                "state": node.get("state"),
                "created_at": node.get("createdAt"),
                "model_pkg": model_pkg,
                "sampler_pkg": sampler_pkg,
                "lookback_hours": lookback_h,
                "resample_freq_hours": resample_h,
                "modalities_choice": modalities_choice,
                "num_modalities": count_modalities(modalities_choice),
                "featurizer_pkg": featurizer_pkg,
                **metric_columns(summary),
            })

    return pd.DataFrame(all_records)


RERUN_QUERY = """
query Reruns($entity: String!, $project: String!, $filters: JSONString!) {
  project(entityName: $entity, name: $project) {
    runs(first: 500, filters: $filters) {
      edges { node { name displayName state createdAt summaryMetrics sweep { name } } }
    }
  }
}
"""


def backfill_reruns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Fills rows without pooled metrics from a finished standalone re-run of the same
    combination (same W&B run name, not part of a sweep, created later). A grid sweep never
    re-issues a combination, so failed ones were re-run outside it (scripts/rerun_sweep_runs.sbatch).
    Only metric columns change; the row keeps its sweep run_id. Returns what was copied."""
    missing = df[df["pooled_auroc"].isna()]
    if missing.empty:
        return []
    headers = get_auth_headers()
    filters = json.dumps({"display_name": {"$in": sorted(set(missing["run_name"]))}})
    edges = post(headers, RERUN_QUERY, {"entity": ENTITY, "project": PROJECT, "filters": filters})["data"]["project"]["runs"]["edges"]
    reruns: Dict[str, Dict[str, Any]] = {}
    for e in edges:
        n = e["node"]
        s = json.loads(n["summaryMetrics"] or "{}")
        if n["state"] != "finished" or n.get("sweep") or s.get("pooled/auroc") is None:
            continue
        if n["displayName"] not in reruns or n["createdAt"] > reruns[n["displayName"]]["createdAt"]:
            reruns[n["displayName"]] = {**n, "summary": s}
    copied = []
    for idx, row in missing.iterrows():
        rr = reruns.get(row["run_name"])
        if rr is None or rr["createdAt"] <= str(row["created_at"]):
            continue
        for k, v in metric_columns(rr["summary"]).items():
            df.at[idx, k] = v
        copied.append({"experiment": row["experiment"], "sweep_run_id": row["run_id"], "run_name": row["run_name"],
                       "rerun_id": rr["name"], "rerun_created": rr["createdAt"], "pooled_auroc": df.at[idx, "pooled_auroc"]})
    return copied


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(SWEEP_SETS), default="standard",
                        help="standard: FLAML + DL + RQ2 featurizer sweeps; foundation: TabPFN / TiRex sweeps")
    parser.add_argument("--out-dir", default=None, help="where to write the CSVs (default depends on --set)")
    parser.add_argument("--backfill-reruns", action="store_true",
                        help="fill runs without pooled metrics from finished standalone re-runs (see backfill_reruns)")
    parser.add_argument("--legacy-columns", action="store_true",
                        help="write only the columns of the 2026-10-07 standard export (drop the later additions)")
    args = parser.parse_args()
    sweeps, default_dir, master_name = SWEEP_SETS[args.set]
    out_dir = Path(args.out_dir or default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_all_sweeps(sweeps)
    if df.empty:
        print("No records retrieved. Please check network/auth.")
        return

    copied = backfill_reruns(df) if args.backfill_reruns else []
    for c in copied:
        print(f"backfilled {c['run_name']} (sweep run {c['sweep_run_id']}) from re-run {c['rerun_id']}: pooled AUROC {c['pooled_auroc']:.3f}")
    if args.legacy_columns:
        df = df.drop(columns=STANDARD_EXTRA_COLUMNS)

    master_path = out_dir / master_name
    df.to_csv(master_path, index=False)
    print(f"\nMASTER RESULTS SAVED TO: {master_path} ({len(df)} rows, {len(df.columns)} columns)")
    for sweep_id, meta in sweeps.items():
        sweep_df = df[df["sweep_id"] == sweep_id]
        sweep_file = out_dir / f"{meta['experiment']}_{sweep_id}.csv"
        sweep_df.to_csv(sweep_file, index=False)
        print(f"Saved {len(sweep_df)} runs for {meta['experiment']} to {sweep_file}")
    if copied:
        pd.DataFrame(copied).to_csv(out_dir / "backfilled_reruns.csv", index=False)
        print(f"Backfill record: {out_dir / 'backfilled_reruns.csv'}")

    print("\n--- Summary by Sweep & State ---")
    print(df.groupby(["sweep_id", "experiment", "state"]).size().unstack(fill_value=0))
    still = df[df["pooled_auroc"].isna()]
    print(f"\nRows without pooled metrics after export: {len(still)}")
    for _, r in still.iterrows():
        print(f"  {r['experiment']} | {r['run_name']} | {r['run_id']} | state {r['state']}")


if __name__ == "__main__":
    main()
