<div align="center">
  
# Callback Configurations

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

</div>

This directory contains configuration files for **PyTorch Lightning Callbacks**. Callbacks allow you to add non-essential logic to your training loop (like checkpointing, early stopping, or custom logging) without cluttering your model code.

## 📌 Available Callbacks

### 1. `default.yaml`
A collection of the most commonly used callbacks. When you run with `callbacks=default`, it typically enables:
- Model Checkpointing
- Early Stopping
- Model Summary
- Rich Progress Bar

### 2. `model_checkpoint.yaml`
Automatically saves the best version of your model based on a monitored metric (usually `val/acc`).
- **Use Case:** Ensuring you don't lose the best weights if the model overfits later.

### 3. `early_stopping.yaml`
Stops training automatically if the validation metric stops improving for a certain number of epochs (patience).
- **Use Case:** Saving compute time and preventing overfitting.

### 4. `label_history.yaml` (Custom)
A specialized utility for longitudinal modeling. It prints the full chronological history of survey answers for a specific participant at the start of the run.
- **Use Case:** Verifying the "Lag" baseline and inspecting raw data sequences.
- **Command:** `python src/train.py callbacks=label_history callbacks.label_history.target_user_id=27`

### 5. `confusion_matrix.yaml` (Custom)
Prints a beautifully formatted confusion matrix to the terminal using the `Rich` library.
- **Use Case:** Visualizing class-level performance and misclassifications directly in the console.
- **Parameters:**
  - `frequency`: How many epochs to skip between prints (e.g., `frequency=5` prints every 5th epoch).
- **Command:** `python src/train.py callbacks=confusion_matrix callbacks.confusion_matrix.frequency=1`

### 6. `classification_metrics.yaml` (Custom)
Logs advanced metrics like **F1-Score** and **AUROC** to your logger (TensorBoard/WandB).
- **Use Case:** Tracking more robust performance indicators than simple accuracy, especially for imbalanced data.
- **Metrics included:** `val/f1`, `val/auroc`, `test/f1`, `test/auroc`.
- **Command:** `python src/train.py callbacks=classification_metrics`

### 7. `rich_progress_bar.yaml`
Replaces the default progress bar with a more detailed, "Rich" version that includes estimated time remaining and metric formatting.

---

## ⚡ Your Superpowers

### Use a specific callback group
```bash
python src/train.py callbacks=default
```

### Enable only one callback
```bash
python src/train.py callbacks=early_stopping
```

### Disable all callbacks
```bash
python src/train.py callbacks=none
```

### Overriding callback parameters
You can tune callback behavior directly from the terminal:
```bash
# Increase early stopping patience to 10 epochs
python src/train.py callbacks=early_stopping callbacks.early_stopping.patience=10

# Save top 3 checkpoints instead of just the best one
python src/train.py callbacks=model_checkpoint callbacks.model_checkpoint.save_top_k=3
```

---

## 🛠️ Creating New Callbacks

1.  Write your callback class in `src/utils/callbacks.py` (inheriting from `lightning.Callback`).
2.  Create a corresponding `.yaml` file in this directory.
3.  Set the `_target_` to point to your new Python class.

**Example `my_callback.yaml`:**
```yaml
my_callback:
  _target_: src.utils.callbacks.MyCallback
  param1: "value"
```
