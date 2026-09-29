import os
import json
import netrc
import requests
import pandas as pd
from pathlib import Path

netrc_path = Path.home() / ".netrc"
auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
headers = {"Authorization": f"Bearer {auth[2]}", "Content-Type": "application/json"}

query = """
query SweepRuns($entity: String!, $project: String!, $name: String!) {
  project(entityName: $entity, name: $project) {
    sweep(sweepName: $name) {
      runs(first: 2000) {
        edges {
          node {
            name
            displayName
            state
            summaryMetrics
            config
            createdAt
          }
        }
      }
    }
  }
}
"""

sweeps = {
    'mod2y6eh': 'sampler_model_sweep (4 tabular models)',
    '7ietfm26': 'dl_sampler_sweep (3 deep learning models)'
}

records = []

for sweep_id, desc in sweeps.items():
    print(f"Fetching sweep {sweep_id}: {desc} via direct GraphQL...")
    resp = requests.post(
        "https://api.wandb.ai/graphql",
        headers=headers,
        json={
            "query": query,
            "variables": {"entity": "hickeykevin", "project": "lemurs-modeling", "name": sweep_id}
        },
        timeout=30
    )
    data = resp.json()
    edges = data["data"]["project"]["sweep"]["runs"]["edges"]
    print(f"Retrieved {len(edges)} runs for {sweep_id}.")
    
    for e in edges:
        node = e["node"]
        summary = json.loads(node.get("summaryMetrics") or "{}")
        config = json.loads(node.get("config") or "{}")
        
        # Unwrap config value dicts if nested {"value": ...}
        def get_val(cfg, key):
            v = cfg.get(key)
            if isinstance(v, dict) and "value" in v:
                return v["value"]
            return v
            
        model_val = get_val(config, "model_pkg")
        if not model_val:
            model_dict = get_val(config, "model")
            if isinstance(model_dict, dict):
                model_val = model_dict.get("name")
        
        sampler_val = get_val(config, "sampler_pkg")
        mod_val = get_val(config, "modalities_choice")
        if not mod_val:
            mod_val = get_val(config, "modalities")
            
        auroc = summary.get("pooled/auroc", summary.get("test/pooled_auroc"))
        f1 = summary.get("pooled/f1", summary.get("test/pooled_f1"))
        prec = summary.get("pooled/precision", summary.get("test/pooled_precision"))
        rec = summary.get("pooled/recall", summary.get("pooled/sensitivity", summary.get("test/pooled_sensitivity")))
        spec = summary.get("pooled/specificity", summary.get("test/pooled_specificity"))
        
        records.append({
            "sweep_id": sweep_id,
            "run_id": node["name"],
            "run_name": node["displayName"],
            "state": node["state"],
            "created_at": node["createdAt"],
            "model_pkg": model_val,
            "sampler_pkg": sampler_val,
            "modalities_raw": json.dumps(mod_val) if isinstance(mod_val, (list, dict)) else str(mod_val),
            "pooled_auroc": auroc,
            "pooled_f1": f1,
            "pooled_precision": prec,
            "pooled_recall": rec,
            "pooled_specificity": spec,
        })

df = pd.DataFrame(records)
os.makedirs("reports/sweep_analysis", exist_ok=True)
out_csv = "reports/sweep_analysis/sweeps_mod2y6eh_7ietfm26.csv"
df.to_csv(out_csv, index=False)

print(f"\nSuccessfully saved {len(df)} records to {out_csv}!")
print("\nRun states by sweep:")
print(df.groupby(["sweep_id", "state"]).size())

print("\nModel counts for finished runs:")
print(df[df["state"] == "finished"].groupby(["sweep_id", "model_pkg"]).size())
