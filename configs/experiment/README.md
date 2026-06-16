<div align="center">

# 🧪 Named Experiment Configurations

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

Experiment configurations represent the highest level of control in the project configuration system. Instead of typing lengthy, error-prone shell overrides, experiment configurations allow you to group multiple component swaps and hyperparameter values into a single version-controlled YAML file.

---

## 🚀 Available Experiment Presets

### 1. `lstm_health.yaml`
Standard deep learning training run configuring the LSTM model, classification metrics, GPU accelerator, and health time-series loaders.
*   **Command:**
    ```bash
    uv run src/train.py experiment=lstm_health
    ```

### 2. `flaml_health.yaml`
AutoML baseline specifically configured for running tabular model optimization over flattened health sequences.
*   **Command:**
    ```bash
    uv run src/train.py experiment=flaml_health
    ```

### 3. `regression_health.yaml`
Deep learning training run configured for regression tasks, utilizing the `HealthRegressionLitModule` to predict continuous target scores and logging regression-specific metrics (like RMSE, MAE, R², etc.) via the `RegressionMetricsCallback`.
*   **Command:**
    ```bash
    uv run src/train.py experiment=regression_health data/aggregator=suicide_risk_regression
    ```

---

## 🛠️ Creating Your Own Custom Experiment

To create a new research run configuration, add a file (e.g. `my_research_run.yaml`) to `configs/experiment/` and follow these rules:

### 1. Set the Package Header
Always place `# @package _global_` at the very top. This tells Hydra to merge your custom properties into the global config root rather than nesting them under a sub-key.

### 2. Set Up Defaults & Overrides
Specify which core config components you want to swap out (using the `/` prefix).

### 3. Define Hyperparameter Specifics
Add your custom parameter parameters after the defaults block.

### Example Template: `my_research_run.yaml`
```yaml
# @package _global_

# 1. Establish the base configs to override
defaults:
  - override /data: health
  - override /model: health
  - override /data/sampler: rolling_hour
  - override /callbacks: [classification_metrics, confusion_matrix]

# 2. Tag and document the run for logging
tags: ["exp_rolling_48h", "adam_optimizer"]
seed: 12345

# 3. Model parameters overrides
model:
  optimizer:
    lr: 0.002
  net:
    hidden_size: 128
    dropout: 0.25

# 4. Data sampling overrides
data:
  batch_size: 32
  sampler:
    lookback_hours: 48.0
    resample_freq: "1h"

# 5. Trainer duration limits
trainer:
  max_epochs: 30
```

---

## ❓ Why Use Experiments vs. Command Line Overrides?

1.  **Reproducibility**: You can commit YAML files to Git to capture the exact parameter landscape of your publication or training run.
2.  **Cleanliness**: Sharing a config file name is much easier than sharing a 600-character terminal command.
3.  **WandB Grouping**: You can map `tags: ${tags}` inside the experiment so that your logs automatically group under the experiment label.
