<div align="center">

# 🚶 Lemurs Modeling Framework

[![python](https://img.shields.io/badge/-Python_3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

*Modeling, feature engineering, and evaluation framework for the Lemurs health study.*

Built with **PyTorch Lightning** and **Hydra** for highly scalable, modular, and reproducible machine learning experiments.

</div>

---

## 📌 Introduction

This repository contains the modeling logic, time-series feature generators, and clinical target binarization utilities for processing raw mobile sensor and ecological momentary assessment (EMA) data.

To make onboarding as smooth as possible for team members new to **PyTorch Lightning** or **Hydra**, this documentation is split into three progressive learning tracks:
* [**🟢 Beginner Track**](#-beginner-track): Environment setup, running defaults, and understanding where files live.
* [**🟡 Intermediate Track**](#-intermediate-track): CLI overrides, swapping modular metrics/samplers/models, and logging.
* [**🔴 Advanced Track**](#-advanced-track): Creating custom components, writing new experiments, running Optuna sweeps, and distributed training.

---

## 📁 Repository Structure

```
├── configs                   <- Hydra configuration YAML files (the "brain")
│   ├── callbacks                <- Logging, checkpointing, and early stopping callbacks
│   ├── data                     <- Datamodule configs (including samplers, aggregators)
│   ├── debug                    <- Troubleshooting configurations
│   ├── experiment               <- Named experiments (override groups)
│   ├── logger                   <- Logging options (WandB, CSV, TensorBoard)
│   ├── model                    <- Model architectures, optimizers, and schedulers
│   ├── trainer                  <- PyTorch Lightning trainer flags (CPU, GPU, epochs)
│   ├── eval.yaml             <- Entry config for model testing
│   └── train.yaml            <- Entry config for model training
│
├── notebooks                 <- Interactive Jupyter / Streamlit dashboards
│
├── scripts                   <- Bash utility scripts (Slurm sweeps, wrappers)
│
├── src                       <- Core Python source code
│   ├── data                     <- Data loading, train-test splits, database extractors
│   │   └── components              <- Time-series samplers and label aggregators
│   ├── models                   <- PyTorch Lightning Modules and Neural Nets
│   │   └── components              <- Concrete models (LSTM, simple networks)
│   ├── utils                    <- Callbacks, helper tools, and rich printing formatters
│   ├── train.py                 <- Main model training execution script
│   └── eval.py                  <- Main model checkpoint evaluation script
│
├── tests                     <- Pytest unit and integration smoke tests
└── pyproject.toml            <- Package dependencies and build configurations
```

---

## 🟢 Beginner Track

If you are new to the repository, start here to set up your environment and run your first model.

### 1. Installation & Environment Setup

We use **uv** to manage packages, virtual environments, and python runtimes. It replaces `pip`, `conda`, and `poetry` with near-instant speed.

1. Install `uv` on your machine:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone this repository and sync dependencies:
   ```bash
   git clone https://github.com/hickeykevin/lemurs-modeling.git
   cd lemurs-modeling
   uv sync
   ```

### 2. Running Your First Training Job

To run a default training run using pre-configured CPU settings:
```bash
uv run src/train.py
```
This command triggers a baseline run that:
1. Connects to the database and extracts steps/sensor sequences.
2. Applies a default daily data aggregator and time sampler.
3. Fits a PyTorch LSTM model.
4. Outputs logs and training telemetry directly to your console.

### 3. Git Workflow & Collaboration Guidelines

To keep the codebase clean and avoid merge conflicts, follow this Git procedure when testing or developing new models, samplers, or visualization features:

1.  **Start with an Up-to-Date Local `main`**
    Before starting work, checkout `main` and pull latest changes:
    ```bash
    git checkout main
    git pull origin main
    ```
2.  **Create a Named Feature Branch**
    Create a new branch off `main` using standard naming conventions:
    *   For new models or architectures: `feat/modeling-your-feature`
    *   For analysis or exploratory visual work: `eda/your-analysis`
    *   For speed improvements or refactoring: `refactor/what-changed`
    ```bash
    git checkout -b feat/modeling-transformer-layers
    ```
3.  **Make Small, Logical Commits**
    Stage only the code and configuration files relevant to your task. **Do not** stage log files (`logs/`), database dumps, or local runtime environments.
    ```bash
    git add src/models/components/transformer.py configs/model/transformer.yaml
    git commit -m "feat(modeling): add self-attention transformer blocks and config"
    ```
4.  **Push and Open a Pull Request**
    Push your local branch to GitHub:
    ```bash
    git push -u origin feat/modeling-transformer-layers
    ```
    Once pushed, navigate to the repository on GitHub and open a Pull Request (PR) for review.

---

### 💡 Step-by-Step Git Walkthrough Example

Here is a practical, end-to-end example of a developer adding a new **transformer baseline** model:

#### Step 1: Initialize the branch
```bash
# Switched to branch 'main'
git checkout main

# Pull latest commits from GitHub
git pull origin main

# Create and switch to a new branch for transformer development
git checkout -b feat/modeling-transformer-baseline
```

#### Step 2: Implement and test locally
Let's say the developer adds a network component at `src/models/components/transformer.py` and a config file at `configs/model/transformer.yaml`. 

Before committing, you **must** run local sanity checks to ensure the new architecture loads and doesn't throw runtime exceptions during training, validation, or metric calculation. Use one of our pre-configured debugging modes:

##### Option A: Fast Dev Run (`debug=fdr`)
If you just want to verify that data flows through your model layers without compile or tensor dimension errors, use:
```bash
uv run src/train.py model=transformer debug=fdr
```
*   **What it does**: Activates PyTorch Lightning's `fast_dev_run=true`, executing exactly $1$ training batch, $1$ validation batch, and $1$ test batch. 
*   **Behavior**: It bypasses early stopping, checkpoint saving, and metric logging callbacks to execute as quickly as possible.

##### Option B: Limited Batch Mock Epochs (`debug=limit`)
If you want to verify that early stopping, checkpointing, custom classification callbacks, and metric logging behave correctly over multiple mock epochs without waiting for a full dataset run:
```bash
uv run src/train.py model=transformer debug=limit callbacks=default
```
*   **What it does**: Restricts training to $3$ epochs using only $1\%$ of the training data and $5\%$ of the validation/test data.
*   **Behavior**: Unlike `fdr`, callbacks and loggers remain fully active.
*   > [!IMPORTANT]
    > **Enable Callbacks Explicitly**: Because `debug/default.yaml` overrides loggers to null, you must explicitly pass `callbacks=default` (or your custom callback config) and `logger=csv` (or your logger config) in the CLI command to test early stopping or checkpointing outputs during `debug=limit` runs.

---

#### Step 3: Inspect changes and stage precisely
Check `git status` to see what changed:
```bash
git status
```
*Output:*
```
On branch feat/modeling-transformer-baseline
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
	modified:   configs/model/transformer.yaml

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	src/models/components/transformer.py
	logs/   <-- IMPORTANT: DO NOT STAGE LOGS!
```

Stage only the source and config files:
```bash
git add src/models/components/transformer.py configs/model/transformer.yaml
```

#### Step 4: Commit and Push
```bash
# Commit the changes with a clear message
git commit -m "feat(modeling): implement transformer baseline network and model config"

# Push the branch to remote GitHub
git push -u origin feat/modeling-transformer-baseline
```

---

### 4. Understanding PyTorch Lightning + Hydra

Instead of writing custom loops for training, validation, and testing, this repository uses **PyTorch Lightning** to organize training phases:

*   **`LightningDataModule`** ([src/data/health_datamodule.py](src/data/health_datamodule.py)): Manages database queries, time-series alignment, splitting users into train/val/test groups, and setting up dataloaders.
*   **`LightningModule`** ([src/models/health_module.py](src/models/health_module.py)): Houses model weights, defines how inputs pass through the model (`forward`), calculates loss, and calls optimizers.

**Hydra** acts as the configuration layer. Instead of hardcoding arguments in Python, objects are instantiated dynamically from YAML files using the `_target_` keyword.

> [!NOTE]
> **What is `_target_`?**
> Inside `configs/model/health.yaml`, you will see:
> ```yaml
> _target_: src.models.health_module.FLAMLHealthModule
> ```
> At runtime, Hydra imports and instantiates the class listed in `_target_` using the arguments defined beneath it. This makes it trivial to swap components without changing code.

### 5. Finding Checkpoints & Logs

Every execution creates a unique folder inside the `logs/` directory grouped by date and time:
```
logs/
└── train/
    └── runs/
        └── YYYY-MM-DD_HH-MM-SS/
            ├── .hydra/                # Saved copy of full composed YAML configurations
            ├── csv/                   # CSV formatted training logs
            └── checkpoints/
                └── last.ckpt          # Model weights checkpoint at the end of training
```

---

## 🟡 Intermediate Track

Once you have successfully executed the default pipeline, you can customize runs, swap modules, and activate logging dashboard integrations.

### 1. Overriding Parameters via the Command Line

Hydra makes it simple to tweak any parameter from your terminal using dot-notation:

```bash
# Override epochs and learning rate
uv run src/train.py trainer.max_epochs=20 model.optimizer.lr=0.001

# Add a parameter not defined in the default YAML file with a '+'
uv run src/train.py +trainer.val_check_interval=0.25
```

### 2. Swapping Modular Components

This repository is built like building blocks. You can swap out models, time samplers, loggers, or aggregators directly in your CLI command:

```bash
# Use traditional AutoML instead of an LSTM
uv run src/train.py model=flaml

# Use the rolling hours sampler instead of the daily sampler
uv run src/train.py data/sampler=rolling_hour

# Use CSV logging instead of console-only logs
uv run src/train.py logger=csv
```

### 3. Exploring the Clinical Targets (Aggregators)

An aggregator translates multiple raw EMA questionnaire answers into a single target label ($0$ or $1$) for classification.

Available clinical targets you can override (`data/aggregator=preset_name`):
*   `suicide_risk`: Targets self-harm and active ideation indicators.
*   `social_stress`: Targets interpersonal conflict.
*   `emotion_regulation`: Targets behavioral coping mechanisms.
*   `minority_stress`: Targets discrimination and minority stress factors.
*   `positive_emotion` / `negative_emotion`: Targets mood valences.

To customize threshold rules on the fly:
```bash
# Make suicide risk criteria stricter (require a score of at least 3 on the first rule group)
uv run src/train.py data/aggregator=suicide_risk "data.aggregator.rules[0].val=3"
```

### 4. Interactive Exploration Dashboards

We provide Streamlit dashboards to visualize how configurations process raw steps:
```bash
# Run the Interactive Sampler Playground
uv run streamlit run notebooks/sampler_dashboard.py
```
This lets you select users, preview the active sampling lookback window, and check the generated feature tensor structure in real-time.

---

## 🔴 Advanced Track

For core developers writing custom features, designing new network components, or scaling sweeps on compute clusters.

### 1. Creating Custom Model Components

To implement a new neural network:
1.  Add your raw PyTorch network to [src/models/components/](src/models/components/).
2.  Define or extend a wrapper `LightningModule` inside [src/models/](src/models/).
3.  Create a matching YAML configuration file in `configs/model/your_model.yaml` containing the `_target_` import path and hyperparameter defaults:
    ```yaml
    _target_: src.models.your_module.YourLitClass
    net:
      _target_: src.models.components.your_net.YourNetworkComponent
      hidden_dim: 128
    ```
4.  Launch training using your new module:
    ```bash
    uv run src/train.py model=your_model
    ```

### 2. Writing Experiment Config Files

If you find yourself running the same complex CLI override combinations over and over, save them as an **Experiment Config** inside `configs/experiment/`.

Here is an example structure of `configs/experiment/lstm_health.yaml`:
```yaml
# @package _global_

defaults:
  - override /data: health
  - override /model: health
  - override /model/net: lstm
  - override /callbacks: [confusion_matrix, classification_metrics]
  - override /trainer: gpu

tags: ["health", "lstm"]
seed: 42

model:
  net:
    hidden_size: 32
    dropout: 0.2

trainer:
  max_epochs: 100
```
Run the experiment with:
```bash
uv run src/train.py experiment=lstm_health
```

### 3. Hyperparameter Sweeps with Optuna

To search for the best learning rates, batch sizes, and model shapes automatically, run a multi-run sweep using the Optuna optimizer plugin:

```bash
uv run src/train.py -m hparams_search=health_optuna experiment=lstm_health
```
Optuna dynamically executes trials based on the search space defined in `configs/hparams_search/health_optuna.yaml` and logs the best trial configuration to your run logs.

### 4. Distributed Training & Cluster Submissions

For large-scale training across multiple GPUs or cluster nodes:
```bash
# Train on 4 GPUs with Distributed Data Parallel (DDP)
uv run src/train.py trainer=ddp trainer.devices=4
```

To submit runs to a **SLURM cluster**, use our wrapper configs or batch submit scripts:
```bash
sbatch scripts/run_sweep.sbatch
```

---

## 💡 Best Practices

*   **Environment Variables**: Save API keys, credentials, or custom database paths in a local `.env` file (copied from `.env.example`). Hydra resolves them dynamically via `${oc.env:VAR_NAME}` configurations.
*   **Automatic Code Formatting**: We enforce formatting standards. Run formatters before committing changes:
    ```bash
    uv run pre-commit run -a
    ```
*   **Tests**: Execute pytest unit tests to check if configuration changes break pipelines:
    ```bash
    uv run pytest
    ```
*   **Version Control for Large Files**: Never commit raw SQL dumps or large model checkpoints to Git. Set up `dvc` to manage large files:
    ```bash
    uv run dvc add data/raw_data.db
    ```
