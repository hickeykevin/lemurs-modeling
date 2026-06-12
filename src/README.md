# Source Directory (`src/`)

This directory contains the core entry scripts for training, validating, and evaluating models, as well as an interactive graphical user interface (GUI) to configure and launch these processes.

## Files Overview

- **[app.py](app.py)**: A Streamlit-based GUI wrapper for visually managing and executing training and evaluation scripts.
- **[train.py](train.py)**: Main entry point for standard PyTorch Lightning training and testing using Hydra.
- **[cv_train.py](cv_train.py)**: Script for K-fold or Leave-One-User-Out Cross-Validation training and metric aggregation.
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

### 3. Evaluation (`eval.py`)
```bash
uv run src/eval.py model=default data=default ckpt_path=/path/to/checkpoint.ckpt
```
