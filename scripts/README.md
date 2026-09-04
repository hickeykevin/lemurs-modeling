# Sweep & Slurm Execution Scripts (`scripts/`)

This directory contains the active scripts used for running and orchestrating the multi-factor hyperparameter sweep across Slurm cluster nodes.

---

## 🚀 Active Scripts

- **[sampler_model_sweep_wrapper.sh](sampler_model_sweep_wrapper.sh)**:
  - Translates W&B sweep agent arguments into sanitized Hydra configuration overrides.
  - Dispatches `src/train.py` using `eval_plan=cyclical` with pinned parameters (`data/scaler=dual`, `purge_hours=120.0`, `use_demographics=true`, `trainer.max_epochs=75`).

- **[parallel_sweep.sbatch](parallel_sweep.sbatch)**:
  - Slurm batch array job script for launching multiple parallel W&B sweep agents across cluster nodes.
  - Automatically loads the project environment (`uv`) and executes sweep agents.

---

## 📋 Quick Start: Launching Sweep Agents

```bash
# Submit parallel sweep workers on Slurm
sbatch scripts/parallel_sweep.sbatch <SWEEP_ID>
```
