<div align="center">

# ⚙️ Hydra Configuration Brain

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This directory houses the **Hydra configurations** for the Lemurs modeling framework. Every parameter—from dataset extraction parameters to model sizes, optimizer learning rates, loggers, and hardware accelerator choices—is defined here in modular YAML configurations.

---

## 📁 Configuration Structure

The configuration directory is organized into folders representing different functional areas of training and evaluation:

| Directory | Purpose | Key Files / Custom Presets |
| :--- | :--- | :--- |
| [**`data/`**](data/) | Slices, queries, and aggregates time-series | Sampler configurations (`sampler/`), clinical rules (`aggregator/`) |
| [**`model/`**](model/) | Neural Network layers and optimization | Flagship LSTM (`health.yaml`), AutoML baselines (`flaml.yaml`), lag benchmarks |
| [**`callbacks/`**](callbacks/) | Plug-and-play metrics and printing tools | F1/AUROC metrics, Rich Confusion Matrix terminal visualizers |
| [**`experiment/`**](experiment/) | Predefined research configs (Named Runs) | `lstm_health.yaml`, `flaml_health.yaml` |
| [**`trainer/`**](trainer/) | PyTorch Lightning trainer flags | CPU training, GPU settings, precision scaling, deterministic modes |
| [**`logger/`**](logger/) | Monitoring dashboard integrations | Weights & Biases (`wandb.yaml`), CSV logger, TensorBoard |
| [**`hparams_search/`**](hparams_search/) | Hyperparameter tuning settings | Optuna grid/random sweep setups |

---

## 💡 How Config Composition Works

Hydra works by **composing** a single configuration object out of multiple small YAML components at execution time. 

### The Defaults List

Look at the top of [configs/train.yaml](train.yaml). You will see the `defaults` block:
```yaml
defaults:
  - _self_
  - data: health.yaml
  - model: health.yaml
  - callbacks: default.yaml
  - logger: null
  - trainer: default.yaml
  - experiment: null
```

This specifies that when you run `python src/train.py`:
1. Default data configurations come from `configs/data/health.yaml`.
2. Default model layers and parameters come from `configs/model/health.yaml`.
3. ...and so on.

> [!IMPORTANT]
> **Order Matters!**
> Sub-configs listed lower in the `defaults` list override any properties set in sub-configs listed above them.

---

## ⚡ Command Line Overrides Cheat Sheet

You do not need to edit YAML files to change behavior. You can override **any** configuration parameter directly from your shell command.

### 1. Swapping Modules (Config Level)
Use `group=file_basename` to swap out entire configuration groups:
```bash
# Swap to AutoML model and set csv logger
uv run src/train.py model=flaml logger=csv

# Swap the data sampler to block sampling
uv run src/train.py data/sampler=block
```

### 2. Overriding Nested Parameters
Use dot-notation to override specific leaf properties inside files:
```bash
# Change parameters nested inside the model configuration
uv run src/train.py model.net.hidden_size=128 model.optimizer.lr=0.005

# Change parameters inside early stopping callbacks
uv run src/train.py callbacks.early_stopping.patience=15
```

### 3. Adding New Fields
If a parameter is not defined in the YAML file at all, append a `+` symbol to register it:
```bash
# Add a run name for WandB logging
uv run src/train.py +logger.wandb.name="custom_lstm_run"
```

---

## 📖 Sub-Guides
For specific details, refer to:
* [Modeling & Architectures Guide](model/README.md)
* [Named Experiments Guide](experiment/README.md)
* [Data Slicers & Samplers Guide](data/sampler/README.md)
* [Clinical Question Aggregators Guide](data/aggregator/README.md)
* [Train Metrics & Callbacks Guide](callbacks/README.md)
