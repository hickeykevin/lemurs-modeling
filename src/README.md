# Source Directory (`src/`)

This directory contains the core entry scripts for training, validating, and evaluating models, as well as an interactive graphical user interface (GUI) to configure and launch these processes.

## Files Overview

- **[app.py](app.py)**: A Streamlit-based GUI wrapper for visually managing and executing training and evaluation scripts.
- **[train.py](train.py)**: Main entry point for standard PyTorch Lightning training and testing using Hydra.
- **[cv_train.py](cv_train.py)**: Script for K-fold or Leave-One-User-Out Cross-Validation training and metric aggregation.
- **[compare_cv_runs.py](compare_cv_runs.py)**: Compares two CV sweeps fold by fold, reporting a paired delta with its interval after verifying the two runs split the same cohort identically.
- **[eval.py](eval.py)**: Evaluation script to test a trained model checkpoint (`.ckpt`) on a dataset.

---

## Lemurs Modeling Hydra GUI (`app.py`)

`app.py` provides an intuitive Streamlit interface that dynamically parses and resolves your YAML configs under the `configs/` directory. It automatically constructs and runs the corresponding terminal commands.

### 1. Installation & Setup

Ensure you have `streamlit` and project dependencies installed. The project uses `uv` for dependency management:

```bash
# Start the Streamlit server locally
uv run streamlit run src/app.py
```

By default, the app starts on `http://localhost:8501`.

### 2. Interface Sections & Optionalities

The GUI is divided into a two-column layout: **Configuration** (left) and **Execution** (right).

#### A. Configuration Column (Left)
- **Execution Mode**: Choose which script to run:
  - **Train**: Executes `src/train.py`
  - **Cross Validate**: Executes `src/cv_train.py`
  - **Evaluate**: Executes `src/eval.py`
- **Execution Environment**:
  - **Local**: Runs the process locally.
  - **Slurm Cluster**: Prepends the command with `srun` resource requests.
- **Slurm Resources Config** *(Only visible if Slurm is selected)*:
  - **Partition** (default: `short`)
  - **CPU Cores (`-c`)** (default: `2`)
  - **GPUs** (default: `0`)
  - **Memory (`--mem`)** (default: `8192` MB)
  - **Time limit (`-t`)** in minutes (default: `120`)
- **Core Groups**: Dynamically populated from subdirectories in the `configs/` folder:
  - **Data**, **Model**, **Trainer**, **Callbacks**, and others.
  - Selecting a config value opens a **📁 settings expander** to visually modify parameters and select nested subgroup configurations.
- **Other Options**:
  - **Seed**: Seed for random number generators.
  - **Checkpoint Path**: Path to a checkpoint file (e.g., when evaluating or resuming).
  - **Custom Overrides**: A text area to write raw Hydra override arguments (one per line, e.g., `trainer.max_epochs=10`).

#### B. Execution Column (Right)
- **Controls**:
  - **Run Button**: Spawns the command in the background.
  - **Stop Button**: Terminates the running subprocess.
- **Generated Command**: Displays the exact command that will be run in real-time as you tweak configurations.
- **Terminal Output**: Displays real-time logging output from the running script.

### 3. Running on Slurm Compute Nodes (via sinteractive)

If you prefer to run both the Streamlit GUI and the training computations on a compute node instead of the login node:

1. **Start an interactive session**:
   ```bash
   sinteractive
   ```
2. **Identify your assigned compute node** from your terminal prompt (e.g., `username@node042:...` -> node is `node042`).
3. **Launch the Streamlit app** on that compute node:
   ```bash
   uv run streamlit run src/app.py
   ```
4. **Tunnel the port** on your local machine:
   ```bash
   ssh -L 8501:node042:8501 username@login.cluster.edu
   ```
5. **Access the GUI** at `http://localhost:8501`.

---

## Entrypoint Scripts Usage

If you prefer running via the command line interface (CLI), you can invoke the scripts directly:

### 1. Training (`train.py`)
```bash
uv run src/train.py model=default data=default trainer=default
```

### 2. Cross Validation (`cv_train.py`)
```bash
uv run src/cv_train.py model=default data=default data.num_folds=5
```

Each run logs one row per (repeat, fold) — the fold's metrics under `fold/*`, plus a fingerprint of the cohort and of the users held out (`cv/cohort_hash`, `cv/test_user_hash`) — followed by the aggregate `*_mean` / `*_ci_low` / `*_ci_high` summary.

### 3. Comparing two sweeps (`compare_cv_runs.py`)
The partition for a given (repeat, fold) depends only on the seed and the cohort, never on the model. So two sweeps with the same `seed`, `data.num_folds`, `data.num_repeats` and `data.*` settings score the identical users in every cell, and can be differenced fold by fold:

```bash
uv run src/cv_train.py model=A seed=7 data.num_folds=5 data.num_repeats=20 logger=csv
uv run src/cv_train.py model=B seed=7 data.num_folds=5 data.num_repeats=20 logger=csv

uv run src/compare_cv_runs.py \
  logs/cv_train/runs/<run_A>/csv/version_0/metrics.csv \
  logs/cv_train/runs/<run_B>/csv/version_0/metrics.csv \
  --metric test/auroc --label-a "model A" --label-b "model B"
```

Pairing cancels fold difficulty, which at this cohort size is the largest source of spread — so it detects gaps that comparing two independent CV intervals cannot. The script verifies the cohorts and the per-fold held-out users match before differencing, and refuses rather than reporting if they don't.

That refusal matters most for **data-setting** comparisons: changing anything that affects sensor-coverage filtering (sampler window, modality set, `collapse_strategy`) drops different responses, so the same fold index holds different people. Either set `require_sensor_data=False` for both runs, pass the intersected user set as `exclude_user_ids` to both, or compare them unpaired.

### 4. Evaluation (`eval.py`)
```bash
uv run src/eval.py model=default data=default ckpt_path=/path/to/checkpoint.ckpt
```

#### Re-using Configuration from a Previous Run
To evaluate a checkpoint using the exact configuration used during its training (to avoid parameter mismatch), you can point Hydra directly to the run's saved config directory using the `--config-dir` and `--config-name` flags:

```bash
uv run src/eval.py \
  --config-dir logs/train/runs/<run_timestamp>/.hydra \
  --config-name config \
  ckpt_path=logs/train/runs/<run_timestamp>/checkpoints/last.ckpt
```

You can append any additional evaluation-time overrides at the end of the command. Since the saved configuration is already composed, you must use the `+` prefix and surround the list in quotes to prevent shell parsing/globbing issues (e.g., `"+callbacks=[classification_metrics,confusion_matrix]"`).
