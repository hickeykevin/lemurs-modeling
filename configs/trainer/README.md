# 📈 PyTorch Lightning Trainer Configurations

The PyTorch Lightning `Trainer` automates all the loop engineering code required for training deep learning models. It manages hardware device allocation, epoch loops, validation checks, precision scaling, backpropagation, and distributed execution.

Under Hydra, the Trainer is instantiated dynamically from modular configuration files defined in this directory.

---

## 📁 Trainer Presets

The following configurations are available and can be swapped using `trainer=preset_name`:

| Config File | Hardware Target | Description |
| :--- | :--- | :--- |
| [**`default.yaml`**](default.yaml) | CPU (baseline) | Base configuration with standard epoch counts and validation intervals. |
| [**`cpu.yaml`**](cpu.yaml) | CPU | Standard CPU execution target. |
| [**`gpu.yaml`**](gpu.yaml) | GPU | Single-GPU execution target. |
| [**`mps.yaml`**](mps.yaml) | Apple Silicon | GPU training for macOS machines utilizing MPS. |
| [**`ddp.yaml`**](ddp.yaml) | Multi-GPU | Distributed Data Parallel training on 4 GPUs with synchronized batch normalization. |
| [**`ddp_sim.yaml`**](ddp_sim.yaml) | Multi-GPU Simulation | Simulates distributed environments on local machines for debugging DDP behaviors. |

---

## ⚙️ Key Config Parameters

The parameters inside the `Trainer` configs map directly to PyTorch Lightning's `Trainer` class arguments:

* **`accelerator`**: The hardware device to run on. Options include `cpu`, `gpu`, `mps` (Apple Silicon), or `auto`.
* **`devices`**: The number of devices to allocate (e.g., `1` or `4`), or a list of specific device IDs.
* **`max_epochs`**: The maximum number of training epochs to run.
* **`min_epochs`**: The minimum number of epochs to run (prevents early stopping from ending training prematurely).
* **`deterministic`**: When set to `True`, PyTorch uses deterministic algorithms where possible. This improves reproducibility at the cost of slight performance overheads.
* **`precision`**: Controls float precision. Can be set to `16-mixed` (mixed precision) for significant training speed-ups and lower memory utilization on modern GPUs.

---

## 💡 CLI Override Examples

To swap the entire trainer configuration:
```bash
# Train on a single GPU
uv run src/train.py trainer=gpu

# Train on multiple GPUs using DDP
uv run src/train.py trainer=ddp
```

To modify specific parameters on the fly using dot-notation:
```bash
# Override the maximum epochs
uv run src/train.py trainer.max_epochs=25

# Force deterministic training
uv run src/train.py trainer.deterministic=True
```
