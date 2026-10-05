#!/usr/bin/env python3
"""Create W&B sweep for RQ2 FLAML Features (step_calorie_distance)."""

import sys
from pathlib import Path
import yaml
import wandb


def main():
    yaml_path = Path("configs/hparams_search/RQ2_flaml_features_step_calorie_distance.yaml")
    if not yaml_path.exists():
        print(f"Error: Config file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading sweep configuration from {yaml_path}...")
    with open(yaml_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    print("Registering sweep with Weights & Biases (entity='hickeykevin', project='lemurs-modeling')...")
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="hickeykevin",
        project="lemurs-modeling",
    )

    print("\n" + "=" * 60)
    print("SUCCESS: Sweep created!")
    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/hickeykevin/lemurs-modeling/sweeps/{sweep_id}")
    print("-" * 60)
    print("To launch parallel workers on Slurm:")
    print(f"  sbatch scripts/parallel_sweep.sbatch {sweep_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
