<div align="center">

# Experiment Configurations

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

Experiment configurations are the highest level of control in this repository. They allow you to bundle multiple overrides (data, model, trainer, callbacks) into a single file to ensure your runs are version-controlled and reproducible.

## 🚀 Available Experiments

### 1. `lstm_suicide_risk.yaml`
Standard LSTM training run targeting suicide risk aggregation with block sampling.

### 2. `flaml_suicide.yaml`
AutoML baseline specifically configured for suicide risk classification.

---

## ⚡ How to use in Experiments

To execute an experiment, use the `experiment` override:
```bash
python src/train.py experiment=lstm_suicide_risk
```

---

## 🛠️ Creating New Experiments

When creating a new experiment, follow these three rules:

### 1. Use the `@package _global_` header
This tells Hydra to merge the parameters in this file with the global configuration tree rather than nesting them under a sub-key.

### 2. Override existing defaults
Use the `defaults` section to swap out whole components (like the model or data config).

### 3. Add specific hyperparameter overrides
Place these after the `defaults` block.

### Example: `new_research_run.yaml`
```yaml
# @package _global_

defaults:
  - override /model: health.yaml
  - override /data: health.yaml
  - override /data/sampler: rolling_hour

# Overrides
tags: ["exp_01", "rolling_window"]
seed: 42

model:
  optimizer:
    lr: 0.001
  net:
    hidden_size: 128

data:
  batch_size: 32
  sampler:
    lookback_hours: 48

trainer:
  max_epochs: 20
```

## 📍 Why use experiments instead of command line overrides?
- **Reproducibility:** You can commit an experiment file to Git and track exactly what parameters were used.
- **Convenience:** Instead of typing 10 different overrides in your terminal, you just type `experiment=name`.
- **Clarity:** It's easier to read a YAML file than a single 500-character shell command.
