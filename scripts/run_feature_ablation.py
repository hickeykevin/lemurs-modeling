#!/usr/bin/env python
"""Runner script for non-activity feature ablations (Conditions 0 through 11).

Evaluates each condition using:
  - eval_plan=cyclical
  - model=flaml_xgboost
  - data.modalities=step_distance
  - data/preprocessors=step_distance

Usage:
    python scripts/run_feature_ablation.py [--dry-run] [--conditions 0 1 2 ...]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


CONDITIONS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "00_pure_sensor_baseline",
        "description": "Pure Sensor Baseline (step + distance only)",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 1,
        "name": "01_age_only",
        "description": "Age Only",
        "overrides": [
            "data.use_demographics=true",
            "data.use_age=true",
            "data.use_gender=false",
            "data.use_lgbt=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 2,
        "name": "02_gender_only",
        "description": "Gender Only",
        "overrides": [
            "data.use_demographics=true",
            "data.use_age=false",
            "data.use_gender=true",
            "data.use_lgbt=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 3,
        "name": "03_lgbt_only",
        "description": "LGBT / Sexual Orientation Only",
        "overrides": [
            "data.use_demographics=true",
            "data.use_age=false",
            "data.use_gender=false",
            "data.use_lgbt=true",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 4,
        "name": "04_all_demographics",
        "description": "All Demographics (Age + Gender + LGBT)",
        "overrides": [
            "data.use_demographics=true",
            "data.use_age=true",
            "data.use_gender=true",
            "data.use_lgbt=true",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 5,
        "name": "05_device_source_only",
        "description": "Device Source Only",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=true",
            "data.use_survey_context=false",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 6,
        "name": "06_survey_prompt_only",
        "description": "Survey Prompt Only (is_morning)",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=true",
            "data.survey_context_mode=morning_only",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 7,
        "name": "07_referent_window_only",
        "description": "Referent Window Only",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=true",
            "data.survey_context_mode=referent_only",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 8,
        "name": "08_full_survey_context",
        "description": "Full Survey Context",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=true",
            "data.survey_context_mode=both",
            "data.use_sleep=false",
        ],
    },
    {
        "id": 9,
        "name": "09_sleep_hours_only",
        "description": "Sleep Hours Only",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=true",
            "data.sleep_feature_mode=hours_only",
        ],
    },
    {
        "id": 10,
        "name": "10_sleep_category_only",
        "description": "Sleep Category Only",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=true",
            "data.sleep_feature_mode=category_only",
        ],
    },
    {
        "id": 11,
        "name": "11_full_sleep",
        "description": "Full Sleep (Hours + Categories)",
        "overrides": [
            "data.use_demographics=false",
            "data.use_device_source=false",
            "data.use_survey_context=false",
            "data.use_sleep=true",
            "data.sleep_feature_mode=both",
        ],
    },
]


ALL_MODALITY_FILES: List[str] = [
    "step.yaml",
    "calorie.yaml",
    "distance.yaml",
    "step_calorie.yaml",
    "step_distance.yaml",
    "calorie_distance.yaml",
    "step_calorie_distance.yaml",
]


def resolve_modality_config(user_input: str) -> Tuple[str, str]:
    """Resolves modality filename or name into (hydra_package_string, filename).
    
    Accepts:
      - 'step_distance.yaml' or 'step_distance'
      - 'all.yaml' or 'all'
    """
    clean = user_input.strip()
    if clean.endswith(".yaml"):
        fname = clean
        pkg = clean[:-5]
    else:
        fname = f"{clean}.yaml"
        pkg = clean
    return pkg, fname


ALL_ROLLING_SAMPLER_FILES: List[str] = [
    "rolling_24h_4h.yaml",
    "rolling_24h_6h.yaml",
    "rolling_24h_8h.yaml",
    "rolling_24h_12h.yaml",
    "rolling_48h_4h.yaml",
    "rolling_48h_6h.yaml",
    "rolling_48h_8h.yaml",
    "rolling_48h_12h.yaml",
    "rolling_72h_4h.yaml",
    "rolling_72h_6h.yaml",
    "rolling_72h_8h.yaml",
    "rolling_72h_12h.yaml",
    "rolling_96h_4h.yaml",
    "rolling_96h_6h.yaml",
    "rolling_96h_8h.yaml",
    "rolling_96h_12h.yaml",
    "rolling_120h_4h.yaml",
    "rolling_120h_6h.yaml",
    "rolling_120h_8h.yaml",
    "rolling_120h_12h.yaml",
]


def resolve_sampler_config(user_input: str) -> Tuple[str, str]:
    """Resolves rolling sampler filename or name into (hydra_package_string, filename).
    
    Accepts only rolling configs (e.g. 'rolling_120h_8h.yaml' or 'rolling_120h_8h').
    Rejects offset and interval samplers.
    """
    clean = user_input.strip()
    if clean.endswith(".yaml"):
        fname = Path(clean).name
        stem = Path(clean).stem
        orig_clean = clean[:-5]
    else:
        fname = f"{Path(clean).name}.yaml"
        stem = Path(clean).name
        orig_clean = clean

    if not fname.startswith("rolling_"):
        raise ValueError(
            f"Invalid sampler '{user_input}'. Only rolling sampler configurations are permitted "
            f"(e.g. rolling_120h_8h.yaml, rolling_24h_4h.yaml). Offset and interval samplers are not allowed."
        )

    sampler_dir = Path(__file__).resolve().parent.parent / "configs" / "data" / "sampler"
    sweep_dir = sampler_dir / "sweep_configs"

    if (sweep_dir / fname).exists():
        return f"sweep_configs/{stem}", fname
    if (sampler_dir / fname).exists():
        return stem, fname
    if orig_clean.startswith("sweep_configs/"):
        return orig_clean, fname
    return f"sweep_configs/{stem}", fname


import concurrent.futures
import re
import time


def _run_single_condition(
    cond: Dict[str, Any],
    mod_pkg: str,
    mod_fname: str,
    sampler_pkg: str,
    sampler_fname: str,
    base_args: List[str],
    cwd: str,
    log_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cond_id = cond["id"]
    name = cond["name"]
    desc = cond["description"]
    overrides = cond["overrides"]

    sampler_stem = sampler_fname[:-5] if sampler_fname.endswith(".yaml") else sampler_fname
    mod_stem = mod_fname[:-5] if mod_fname.endswith(".yaml") else mod_fname
    task_name = f"ablation_{sampler_stem}_{mod_stem}_{name}"

    cmd = [
        sys.executable,
        "src/train.py",
        f"task_name={task_name}",
        f"data/sampler={sampler_pkg}",
        f"data/modalities={mod_pkg}",
    ] + base_args + overrides

    cond_log_dir = log_dir / sampler_stem / mod_stem
    cond_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = cond_log_dir / f"{name}.log"

    result_entry = {
        "sampler": sampler_fname,
        "modality": mod_fname,
        "id": cond_id,
        "name": name,
        "description": desc,
        "status": "PENDING",
        "duration_s": 0.0,
        "auroc": None,
        "auprc": None,
        "f1": None,
        "balanced_acc": None,
        "log_file": str(log_file),
    }

    print(f"\n[STARTING] Sampler: {sampler_fname} | Modality: {mod_fname} | Condition {cond_id}: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}\n")

    if dry_run:
        result_entry["status"] = "DRY_RUN"
        return result_entry

    start_time = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lf.write(proc.stdout)

        elapsed = time.time() - start_time
        result_entry["duration_s"] = round(elapsed, 1)

        # Parse metrics from output
        auroc_match = re.search(r"Pooled AUROC:\s+([0-9\.]+)", proc.stdout)
        auprc_match = re.search(r"Pooled AUPRC:\s+([0-9\.]+)", proc.stdout)
        f1_match = re.search(r"Pooled f1:\s+([0-9\.]+)", proc.stdout)
        bacc_match = re.search(r"Pooled balanced_accuracy:\s+([0-9\.]+)", proc.stdout)

        if auroc_match:
            result_entry["auroc"] = float(auroc_match.group(1))
        if auprc_match:
            result_entry["auprc"] = float(auprc_match.group(1))
        if f1_match:
            result_entry["f1"] = float(f1_match.group(1))
        if bacc_match:
            result_entry["balanced_acc"] = float(bacc_match.group(1))

        if proc.returncode == 0:
            result_entry["status"] = "SUCCESS"
            print(
                f"[FINISHED] Sampler: {sampler_fname} | Modality: {mod_fname} | Condition {cond_id} ({name}) in {elapsed/60:.1f}m -- "
                f"AUROC: {result_entry['auroc']}, AUPRC: {result_entry['auprc']}, F1: {result_entry['f1']}"
            )
        else:
            result_entry["status"] = "FAILED"
            print(f"[ERROR] Sampler: {sampler_fname} | Modality: {mod_fname} | Condition {cond_id} ({name}) failed with returncode {proc.returncode}. See log: {log_file}")

    except Exception as e:
        result_entry["status"] = f"EXCEPTION: {e}"
        print(f"[ERROR] Sampler: {sampler_fname} | Modality: {mod_fname} | Condition {cond_id} ({name}) exception: {e}")

    return result_entry


def run_ablation(
    selected_ids: Optional[List[int]],
    model: str = "flaml_xgboost",
    samplers: Any = "rolling_120h_8h.yaml",
    modalities: Any = "step_distance.yaml",
    featurizer: Optional[str] = None,
    dry_run: bool = False,
    parallel: int = 1,
    output_dir: str = "reports/feature_ablation",
    extra_args: Optional[List[str]] = None,
):
    out_path = Path(output_dir)
    log_dir = out_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Normalize samplers list
    if samplers == "all" or (isinstance(samplers, list) and "all" in samplers):
        samplers_list = ALL_ROLLING_SAMPLER_FILES
    elif isinstance(samplers, str):
        samplers_list = [samplers]
    else:
        samplers_list = list(samplers)
    resolved_samplers = [resolve_sampler_config(s) for s in samplers_list]

    # Normalize modalities list
    if modalities == "all" or (isinstance(modalities, list) and "all" in modalities):
        mod_list = ALL_MODALITY_FILES
    elif isinstance(modalities, str):
        mod_list = [modalities]
    else:
        mod_list = list(modalities)
    resolved_modalities = [resolve_modality_config(m) for m in mod_list]

    base_args = [
        "eval_plan=cyclical",
        f"model={model}",
        "data/scaler=dual",
        "data.os_filter=both",
        "data.collapse_strategy=none",
        "data.require_sensor_data=true",
        "trainer.max_epochs=75",
        "callbacks.model_checkpoint=null",
        "+trainer.enable_checkpointing=false",
        "hydra.job_logging.handlers.file.filename=/dev/null",
        "paths.log_dir=/tmp/khickey/logs",
        'model.automl_config.log_file_name=""',
    ]

    if featurizer is not None:
        base_args.append(f"model/featurizer={featurizer}")

    if extra_args:
        base_args.extend(extra_args)

    target_conditions = [
        cond for cond in CONDITIONS
        if not selected_ids or cond["id"] in selected_ids
    ]

    # Build queue of (condition, (mod_pkg, mod_fname), (sampler_pkg, sampler_fname)) tuples
    jobs = []
    for s_pkg, s_fname in resolved_samplers:
        for m_pkg, m_fname in resolved_modalities:
            for cond in target_conditions:
                jobs.append((cond, m_pkg, m_fname, s_pkg, s_fname))

    cwd = str(Path(__file__).resolve().parent.parent)
    results = []

    print(
        f"\nQueueing {len(jobs)} run(s) across {len(resolved_samplers)} sampler(s) "
        f"and {len(resolved_modalities)} modality/modalities with parallel={parallel}..."
    )

    if parallel > 1 and not dry_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_job = {
                executor.submit(
                    _run_single_condition, cond, m_pkg, m_fname, s_pkg, s_fname, base_args, cwd, log_dir, dry_run
                ): (cond, m_fname, s_fname)
                for cond, m_pkg, m_fname, s_pkg, s_fname in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                res = future.result()
                results.append(res)
    else:
        for cond, m_pkg, m_fname, s_pkg, s_fname in jobs:
            res = _run_single_condition(cond, m_pkg, m_fname, s_pkg, s_fname, base_args, cwd, log_dir, dry_run)
            results.append(res)

    # Merge with existing summary.csv if present
    csv_file = out_path / "summary.csv"
    if not dry_run and results:
        new_df = pd.DataFrame(results)
        if csv_file.exists():
            try:
                existing_df = pd.read_csv(csv_file)
                if "sampler" not in existing_df.columns:
                    existing_df["sampler"] = "rolling_120h_8h.yaml"
                if "modality" not in existing_df.columns:
                    existing_df["modality"] = "step_distance.yaml"

                key = (
                    existing_df["sampler"].astype(str)
                    + "_"
                    + existing_df["modality"].astype(str)
                    + "_"
                    + existing_df["id"].astype(str)
                )
                new_keys = (
                    new_df["sampler"].astype(str)
                    + "_"
                    + new_df["modality"].astype(str)
                    + "_"
                    + new_df["id"].astype(str)
                )
                merged_df = pd.concat([existing_df[~key.isin(new_keys)], new_df], ignore_index=True)
                merged_df = merged_df.sort_values(["sampler", "modality", "id"]).reset_index(drop=True)
            except Exception:
                merged_df = new_df
        else:
            merged_df = new_df

        cols = [
            "sampler",
            "modality",
            "id",
            "name",
            "description",
            "status",
            "duration_s",
            "auroc",
            "auprc",
            "f1",
            "balanced_acc",
            "log_file",
        ]
        ordered_cols = [c for c in cols if c in merged_df.columns] + [c for c in merged_df.columns if c not in cols]
        merged_df = merged_df[ordered_cols]
        merged_df.to_csv(csv_file, index=False)
        print(f"Saved full summary CSV to {csv_file}\n")
        display_results = merged_df.to_dict(orient="records")
    else:
        display_results = results

    # Output Summary Table
    print("\n" + "=" * 145)
    print(
        f"{'Sampler':<24} | {'Modality':<24} | {'ID':<3} | {'Condition Name':<28} | {'Status':<8} | {'AUROC':<7} | {'AUPRC':<7} | {'F1':<7} | {'Bal Acc':<7} | {'Time (s)':<8}"
    )
    print("-" * 145)
    for r in display_results:
        s_s = str(r.get("sampler", ""))
        m_s = str(r.get("modality", ""))
        auroc_s = f"{r['auroc']:.4f}" if pd.notna(r.get("auroc")) else "N/A"
        auprc_s = f"{r['auprc']:.4f}" if pd.notna(r.get("auprc")) else "N/A"
        f1_s = f"{r['f1']:.4f}" if pd.notna(r.get("f1")) else "N/A"
        bacc_s = f"{r['balanced_acc']:.4f}" if pd.notna(r.get("balanced_acc")) else "N/A"
        dur_s = f"{r['duration_s']:.1f}" if pd.notna(r.get("duration_s")) else "0.0"
        print(
            f"{s_s:<24} | {m_s:<24} | {int(r['id']):<3} | {str(r['name']):<28} | {str(r['status']):<8} | {auroc_s:<7} | {auprc_s:<7} | {f1_s:<7} | {bacc_s:<7} | {dur_s:<8}"
        )
    print("=" * 145 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run non-activity feature ablations (Conditions 0-11) across sampler and modality configurations"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--conditions", type=int, nargs="+", help="Specific condition IDs to run (0-11)")
    parser.add_argument(
        "--parallel", "-j", type=int, default=1, help="Number of conditions to run concurrently (e.g. 2, 3, 4)"
    )
    parser.add_argument("--model", type=str, default="flaml_xgboost", help="Model config (e.g. flaml_xgboost, flaml_svm)")
    parser.add_argument(
        "--sampler",
        "--samplers",
        type=str,
        nargs="+",
        default=["rolling_120h_8h.yaml"],
        help=(
            "Rolling sampler config YAML file name(s) (e.g. rolling_120h_8h.yaml, rolling_24h_4h.yaml), "
            "or 'all' for all 20 rolling configs. Note: Only rolling samplers are allowed."
        ),
    )
    parser.add_argument(
        "--all-samplers",
        action="store_true",
        help="Shortcut to run across all 20 rolling sampler YAML configurations",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["step_distance.yaml"],
        help=(
            "Modality config YAML file name(s) (e.g. step.yaml, distance.yaml, calorie.yaml, "
            "step_distance.yaml, step_calorie.yaml, calorie_distance.yaml, step_calorie_distance.yaml), "
            "or 'all' for all 7"
        ),
    )
    parser.add_argument("--all-modalities", action="store_true", help="Shortcut to run across all 7 modality YAML configurations")
    parser.add_argument("--featurizer", type=str, default=None, help="Featurizer override (e.g. 'null')")
    parser.add_argument("--output-dir", type=str, default="reports/feature_ablation", help="Directory for logs and summary CSV")
    parser.add_argument("--extra-args", type=str, nargs="*", default=None, help="Additional Hydra overrides")
    args = parser.parse_args()

    samplers_arg = "all" if args.all_samplers else args.sampler
    modalities_arg = "all" if args.all_modalities else args.modalities

    run_ablation(
        args.conditions,
        model=args.model,
        samplers=samplers_arg,
        modalities=modalities_arg,
        featurizer=args.featurizer,
        dry_run=args.dry_run,
        parallel=args.parallel,
        output_dir=args.output_dir,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    main()


