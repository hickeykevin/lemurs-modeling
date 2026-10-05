"""Gather, extract, and consolidate results from W&B sweeps:
- 9dvq8uiy: RQ1 FLAML Models (560 combinations)
- a6sl4omd: RQ1 DL Models (LSTM, Transformer) (280 combinations)
- zoznzd8y: RQ2 FLAML Features (SummaryStats vs Catch22) (160 combinations)

Saves consolidated CSV and per-sweep CSVs to reports/sweep_results/.
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
    },
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


def fetch_all_sweeps() -> pd.DataFrame:
    headers = get_auth_headers()
    all_records = []

    for sweep_id, meta in SWEEPS.items():
        print(f"\nFetching sweep '{sweep_id}' ({meta['experiment']})...")
        resp = requests.post(
            "https://api.wandb.ai/graphql",
            headers=headers,
            json={
                "query": GRAPHQL_QUERY,
                "variables": {"entity": ENTITY, "project": PROJECT, "name": sweep_id},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        sweep_data = data.get("data", {}).get("project", {}).get("sweep")
        if not sweep_data:
            print(f"Warning: Sweep '{sweep_id}' not found in W&B project {ENTITY}/{PROJECT}")
            continue

        sweep_state = sweep_data.get("state")
        edges = sweep_data.get("runs", {}).get("edges", [])
        print(f"  -> Sweep State: {sweep_state} | Total runs fetched: {len(edges)}")

        for edge in edges:
            node = edge["node"]
            run_id = node.get("name")
            display_name = node.get("displayName")
            state = node.get("state")
            created_at = node.get("createdAt")

            raw_summary = node.get("summaryMetrics") or "{}"
            raw_config = node.get("config") or "{}"
            summary = json.loads(raw_summary) if isinstance(raw_summary, str) else raw_summary
            config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config

            # Extract configs
            sampler_pkg = unwrap_val(config, "sampler_pkg")
            model_pkg = unwrap_val(config, "model_pkg")
            modalities_choice = unwrap_val(config, "modalities_choice")
            featurizer_pkg = unwrap_val(config, "featurizer_pkg", meta["default_featurizer"])

            # Fallbacks if nested in Hydra keys
            if not modalities_choice:
                # In RQ2, modality is fixed to step_distance
                if sweep_id == "zoznzd8y":
                    modalities_choice = "step_distance"
                else:
                    mod_cfg = unwrap_val(config, "data/modalities") or unwrap_val(config, "data.modalities")
                    modalities_choice = str(mod_cfg) if mod_cfg else None

            lookback_h, resample_h = parse_sampler(sampler_pkg)
            n_modalities = count_modalities(modalities_choice)

            # Key evaluation metrics
            auroc = summary.get("pooled/auroc", summary.get("test/pooled_auroc"))
            f1 = summary.get("pooled/f1", summary.get("test/pooled_f1"))
            precision = summary.get("pooled/precision", summary.get("test/pooled_precision"))
            recall = summary.get("pooled/recall", summary.get("pooled/sensitivity", summary.get("test/pooled_sensitivity")))
            specificity = summary.get("pooled/specificity", summary.get("test/pooled_specificity"))
            sens_at_spec = summary.get("pooled/sensitivity_at_spec_0.9", summary.get("test/pooled_sensitivity_at_spec_0.9"))
            
            # Balanced accuracy
            balanced_accuracy = summary.get("pooled/balanced_accuracy", summary.get("test/pooled_balanced_accuracy"))
            if balanced_accuracy is None and recall is not None and specificity is not None:
                balanced_accuracy = (float(recall) + float(specificity)) / 2.0

            # Within person
            wp_auroc_mean = summary.get("within_person/auroc_mean", summary.get("test/within_person_auroc_mean"))
            wp_auroc_median = summary.get("within_person/auroc_median", summary.get("test/within_person_auroc_median"))

            # Fold means
            fold_auroc_mean = summary.get("fold/test/auroc_mean")
            fold_f1_mean = summary.get("fold/test/f1_mean")
            fold_spec_mean = summary.get("fold/test/specificity_mean")

            # Bootstrap CIs if present
            auroc_ci_low = summary.get("pooled/auroc_ci_low")
            auroc_ci_high = summary.get("pooled/auroc_ci_high")

            all_records.append({
                "sweep_id": sweep_id,
                "experiment": meta["experiment"],
                "model_type": meta["model_type"],
                "run_id": run_id,
                "run_name": display_name,
                "state": state,
                "created_at": created_at,
                "model_pkg": model_pkg,
                "sampler_pkg": sampler_pkg,
                "lookback_hours": lookback_h,
                "resample_freq_hours": resample_h,
                "modalities_choice": modalities_choice,
                "num_modalities": n_modalities,
                "featurizer_pkg": featurizer_pkg,
                "pooled_auroc": auroc,
                "pooled_f1": f1,
                "pooled_precision": precision,
                "pooled_recall": recall,
                "pooled_specificity": specificity,
                "pooled_sens_at_spec_0.9": sens_at_spec,
                "pooled_balanced_accuracy": balanced_accuracy,
                "within_person_auroc_mean": wp_auroc_mean,
                "within_person_auroc_median": wp_auroc_median,
                "fold_test_auroc_mean": fold_auroc_mean,
                "fold_test_f1_mean": fold_f1_mean,
                "fold_test_spec_mean": fold_spec_mean,
                "pooled_auroc_ci_low": auroc_ci_low,
                "pooled_auroc_ci_high": auroc_ci_high,
            })

    df = pd.DataFrame(all_records)
    return df


def main():
    out_dir = Path("reports/sweep_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_all_sweeps()

    if df.empty:
        print("No records retrieved. Please check network/auth.")
        return

    # Master consolidated CSV
    master_path = out_dir / "all_sweeps_results.csv"
    df.to_csv(master_path, index=False)
    print(f"\n=======================================================")
    print(f"MASTER RESULTS SAVED TO: {master_path}")
    print(f"Total rows: {len(df)}")
    print(f"=======================================================\n")

    # Per-sweep CSVs
    for sweep_id, meta in SWEEPS.items():
        sweep_df = df[df["sweep_id"] == sweep_id]
        sweep_file = out_dir / f"{meta['experiment']}_{sweep_id}.csv"
        sweep_df.to_csv(sweep_file, index=False)
        print(f"Saved {len(sweep_df)} runs for {meta['experiment']} to {sweep_file}")

    # Print summary statistics table
    print("\n--- Summary by Sweep & State ---")
    summary_table = df.groupby(["sweep_id", "experiment", "state"]).size().unstack(fill_value=0)
    print(summary_table)

    # Print top 5 models per sweep by pooled_auroc
    print("\n--- Top 3 Runs per Sweep by Pooled AUROC ---")
    finished_df = df[df["pooled_auroc"].notnull()]
    for sweep_id, group in finished_df.groupby("sweep_id"):
        top = group.sort_values(by="pooled_auroc", ascending=False).head(3)
        print(f"\nSweep {sweep_id} ({top.iloc[0]['experiment']}):")
        for _, r in top.iterrows():
            print(f"  AUROC={r['pooled_auroc']:.4f} | Model={r['model_pkg']} | Sampler={r['sampler_pkg']} | Modalities={r['modalities_choice']} | Featurizer={r['featurizer_pkg']}")


if __name__ == "__main__":
    main()
