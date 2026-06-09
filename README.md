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

To make onboarding as smooth as possible for team members new to **PyTorch Lightning** or **Hydra**, this documentation is organized into two practical walkthrough tutorials and reference sub-guides:
* [**🚶 Tutorial 1: Getting Started and Local Runs**](#-tutorial-1-getting-started-and-local-runs): Covers environment installation, executing standard baseline models, inspecting local logs/checkpoints, and clean-up.
* [**⚡ Tutorial 2: Creating a Custom Transformer Module & GPU Training**](#-tutorial-2-creating-a-custom-transformer-module--gpu-training): Teaches you how to build a new network component from scratch, write its YAML configuration, target GPU hardware, and master command-line overrides.
* [**⚙️ Advanced Customization & Reference**](#-advanced-customization--reference): Deep-dives into clinical aggregators, visualization playgrounds, hyperparameter tuning sweeps, and SLURM submissions.

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

For more detailed guides on core modules, refer to:
* ⚙️ [configs/README.md](configs/README.md) - Hydra configuration brain
* 📊 [configs/data/README.md](configs/data/README.md) - Data presets, samplers & aggregators
* 📈 [configs/trainer/README.md](configs/trainer/README.md) - Trainer devices & epoch limits
* 📊 [src/data/README.md](src/data/README.md) - Pipeline architecture & DataModules
* 🧠 [src/models/README.md](src/models/README.md) - Class hierarchy & Neural Networks
* 🛠️ [src/utils/README.md](src/utils/README.md) - Helper utilities & Callbacks

---

## 🚶 Tutorial 1: Getting Started and Local Runs

Follow this walkthrough to learn how to check out a branch, run model training locally, inspect generated logs, override parameters, and safely submit your code.

### Step 1: Installation & Setup
First, install **uv** (our package manager), clone the repository, and sync virtual environment dependencies:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync environment
git clone https://github.com/hickeykevin/lemurs-modeling.git
cd lemurs-modeling
uv sync
```

Next, set up your local environment configurations by copying the `.env.example` file:
```bash
cp .env.example .env
```
Open the `.env` file in your editor and fill in the required connection settings.
> [!IMPORTANT]
> **Database Credentials**: To connect to the training databases, you need access credentials. If you do not have these keys, ask the project **Administrator** for the required connection credentials.

### Step 2: Initialize Your Branch
Before modifying any configuration files or writing code, create a dedicated branch off an up-to-date local `main`:
```bash
git checkout main
git pull origin main
git checkout -b feat/modeling-my-first-run
```

### Step 3: Run the Default Baseline
Verify your setup by launching the baseline configuration on your CPU:
```bash
uv run src/train.py
```
This command triggers a baseline run that connects to the database, extracts steps/sensor sequences, pairs them with daily survey answers, compiles them via standard splits, and trains a PyTorch LSTM classifier model.

### Step 4: Inspect the Logs and Checkpoints
By default, the CSV logger is enabled in [configs/train.yaml](configs/train.yaml). Every training execution creates a unique, timestamped folder under `logs/`:
```
logs/
└── train/
    └── runs/
        └── YYYY-MM-DD_HH-MM-SS/      <- Your unique run folder
            ├── .hydra/               # Complete composed YAML configurations
            │   ├── config.yaml       # Full parameter dictionary used
            │   ├── overrides.yaml    # Command line arguments passed
            │   └── hydra.yaml        # Hydra internal runtime variables
            ├── csv/                  # CSV formatted training progress logs
            │   └── metrics.csv       # Loss and accuracy per epoch
            └── checkpoints/          
                └── last.ckpt          # Saved model weights at the end of training
```

### Step 5: Overriding Hyperparameters (Creating a New Run)
Instead of modifying YAML configuration files on disk, you can override any setting directly from your command line. For example, to run a quick test with a modified learning rate and epoch count:
```bash
uv run src/train.py model.optimizer.lr=0.01 trainer.max_epochs=5
```
Executing this command creates a **new**, separate timestamped folder under `logs/train/runs/` tracking this specific run's outputs. Refer to [configs/model/README.md](configs/model/README.md) and [configs/README.md](configs/README.md) for more details on available models and settings.

### Step 6: Close Out the Branch
Since we were just running local experiments and did not make any source code changes, we do not need to commit anything. We can simply clean up by discarding any local configuration changes, returning to `main`, and deleting our temporary branch. 

*(Note: The procedure for staging, committing, and pushing actual code modifications is detailed in Step 6 of **Tutorial 2**).*

```bash
# Return to the main branch
git checkout main

# Delete the temporary branch
git branch -D feat/modeling-my-first-run
```

---

## ⚡ Tutorial 2: Creating a Custom Transformer Module & GPU Training

Follow this walkthrough to implement a new network component, define its configuration, launch it using GPU acceleration, and learn how to manage Hydra parameters.

### Step 1: Initialize Your Feature Branch
Start by checking out a clean branch off `main`:
```bash
git checkout main
git pull origin main
git checkout -b feat/modeling-transformer-baseline
```

### Step 2: Implement the PyTorch Network
Assume we create a custom network class (e.g., a self-attention model) inside `src/models/components/transformer.py`:
```python
import torch
import torch.nn as nn

class TransformerComponent(nn.Module):
    # Your custom PyTorch layers go here...
    ...
```

### Step 3: Create the Config File
Based on the "new" model we just made, create the associated configuration file at `configs/model/transformer.yaml`. This file defines the `_target_` import path and hyperparameter defaults (for more details on config composition and structures, see [configs/model/README.md](configs/model/README.md) and [configs/README.md](configs/README.md)):
```yaml
_target_: src.models.health_module.HealthLitModule
net:
  _target_: src.models.components.transformer.TransformerComponent
  hidden_dim: 128
  num_heads: 4
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

### Step 4: Run Your Model on GPU
We then can launch training using the newly defined model configuration and swap the trainer to use GPU acceleration (if available, else run `trainer=cpu`):
```bash
uv run src/train.py model=transformer trainer=gpu
```

### Step 5: Parameter Overrides vs Config Groups (CLI Cheat Sheet)
When overriding configurations from your terminal, distinguish between **structural changes** (swapping full configs) and **value changes** (leaf parameters):

1. **Config Groups (Structural Choices)**
   Swaps an entire module or configuration file. Use the group directory name and file basename (without `.yaml`):
   * `model=transformer` (swaps model to use `configs/model/transformer.yaml`)
   * `trainer=gpu` (swaps hardware trainer settings to use `configs/trainer/gpu.yaml`)
   * `data/sampler=rolling_hour` (swaps data sampler to use `configs/data/sampler/rolling_hour.yaml`)

2. **Simple Parameters (Nested Leaf Parameters)**
   Directly alters a specific value inside a YAML config file using dot-notation:
   * `model.net.hidden_dim=256` (changes network hidden dimension)
   * `model.optimizer.lr=0.005` (changes optimizer learning rate)
   * `trainer.max_epochs=20` (changes maximum epochs)
   * `+trainer.val_check_interval=0.25` (adds a new configuration field with a `+` prefix)

---

## ⚙️ Advanced Customization & Reference

### 1. Exploring Clinical Targets (Aggregators)
An aggregator translates multiple raw EMA questionnaire answers into a single target label (0 or 1) for classification.

Available clinical targets you can override (`data/aggregator=preset_name`):
* `suicide_risk`: Targets self-harm and active ideation indicators.
* `social_stress`: Targets interpersonal conflict.
* `emotion_regulation`: Targets behavioral coping mechanisms.
* `minority_stress`: Targets discrimination and minority stress factors.
* `positive_emotion` / `negative_emotion`: Targets mood valences.

To customize threshold rules on the fly:
```bash
# Make suicide risk criteria stricter (require a score of at least 3 on the first rule group)
uv run src/train.py data/aggregator=suicide_risk "data.aggregator.rules[0].val=3"
```

### 2. Interactive Exploration Dashboards
We provide Streamlit dashboards to visualize how configurations process raw steps:
```bash
# Run the Interactive Sampler Playground
uv run streamlit run notebooks/sampler_dashboard.py
```
This lets you select users, preview the active sampling lookback window, and check the generated feature tensor structure in real-time.

### 3. Writing Experiment Config Files
Save complex CLI override combinations as an **Experiment Config** inside `configs/experiment/`.

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

### 4. Hyperparameter Sweeps with Optuna
To search for the best learning rates, batch sizes, and model shapes automatically, run a multi-run sweep using the Optuna optimizer plugin:
```bash
uv run src/train.py -m hparams_search=health_optuna experiment=lstm_health
```
Optuna dynamically executes trials based on the search space defined in `configs/hparams_search/health_optuna.yaml` and logs the best trial configuration to your run logs.

### 5. Distributed Training & Cluster Submissions
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
