<div align="center">

# Hydra Configuration System

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This directory is the brain of the repository. Every aspect of the data loading, model architecture, training loop, and logging is controlled by these YAML files.

## 📁 Directory Structure

| Folder | Description |
| :--- | :--- |
| [**`data/`**](data/) | DataModule, Samplers, and Label Aggregators. |
| [**`model/`**](model/) | Model architectures (LSTM, Baseline, AutoML) and optimizers. |
| [**`callbacks/`**](callbacks/) | Checkpointing, early stopping, and custom hooks. |
| [**`experiment/`**](experiment/) | High-level configs that override multiple parameters for specific runs. |
| [**`trainer/`**](trainer/) | PyTorch Lightning Trainer flags (GPU, epochs, precision). |
| [**`logger/`**](logger/) | Logging integrations (WandB, Tensorboard, CSV). |
| [**`hparams_search/`**](hparams_search/) | Hyperparameter optimization settings (Optuna). |

---

## 🚀 Key Files

### 1. `train.yaml`
The primary entry point for training. It defines the default "composition" of your experiment. If you run `python src/train.py` without arguments, it uses the defaults specified here.

### 2. `eval.yaml`
Used for evaluating a trained model checkpoint on a test set. It mirrors `train.yaml` but disables training-specific logic.

---

## ⚡ Your Superpowers

### The Composition Principle
Hydra builds your config by stacking these files. The order of `defaults` in `train.yaml` determines who wins in a conflict.

### Command Line Overrides
You can override **anything** from the terminal. 

```bash
# Basic swap of a module
python src/train.py model=lag data/sampler=lag

# Deep override of a nested parameter
python src/train.py model.net.hidden_size=256 trainer.max_epochs=50
```

### Experiment Versioning
Instead of typing long commands, save them as an **Experiment Config** in `configs/experiment/`.
```bash
# Runs a pre-defined set of overrides
python src/train.py experiment=baseline_comparison
```

---

## 📖 Deep Dives
For more specific instructions, visit the READMEs in the subdirectories:
- [Data Samplers & Aggregators Guide](data/sampler/README.md)
- [Modeling & Baselines Guide](model/README.md)
- [Callbacks & Utilities Guide](callbacks/README.md)
