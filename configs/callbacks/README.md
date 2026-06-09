<div align="center">

# 🎛️ PyTorch Lightning Callbacks

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

</div>

Callbacks allow you to inject custom code at specific hooks in the training pipeline (e.g., `on_train_epoch_end`, `on_validation_batch_start`) without modifying the core neural network model or training loops.

---

## 📌 Available Callbacks

### 1. `default.yaml`
Combines the most standard helper tools for production modeling. Enabling `callbacks=default` turns on:
*   **Model Checkpointing**: Save model weights.
*   **Early Stopping**: Halt if validation stalls.
*   **Progress Bars**: Console progress monitoring.

### 2. `model_checkpoint.yaml`
Saves your model parameters based on evaluation metrics.
*   **Monitored Metric**: Defaults to `val/acc` or custom validation target.
*   **Purpose**: Prevents weight loss from training overfitting.

### 3. `early_stopping.yaml`
Stops training run if a monitored metric has not improved for a designated number of epochs (`patience`).
*   **Purpose**: Saves computational budget.

### 4. `confusion_matrix.yaml` (Custom)
Prints a beautifully structured, terminal-friendly confusion matrix at the end of validation epochs using the Python `Rich` package.
*   **Command:**
    ```bash
    uv run src/train.py callbacks=[confusion_matrix]
    ```

### 5. `classification_metrics.yaml` (Custom)
Automatically logs clinical validation metrics—including **F1-Score** and **AUROC**—to your logging dashboard.
*   **Metrics logged**: `val/f1`, `val/auroc`, `test/f1`, `test/auroc`, `val/f1_best`, `val/auroc_best`.
*   **Command:**
    ```bash
    uv run src/train.py callbacks=[classification_metrics]
    ```

### 6. `label_history.yaml` (Custom)
A logging inspector that prints out a participant's longitudinal EMA response history at the beginning of the run. Useful for validating sequential sequence window slicing.
*   **Command:**
    ```bash
    # Inspect user 27's label sequence
    uv run src/train.py callbacks=[label_history] callbacks.label_history.target_user_id=27
    ```

---

## ⚡ Customizing Callbacks from CLI

You can configure callback parameters directly:

```bash
# Set early stopping patience to 10 epochs
uv run src/train.py callbacks=default callbacks.early_stopping.patience=10

# Save top 3 model checkpoints instead of just the best 1
uv run src/train.py callbacks=default callbacks.model_checkpoint.save_top_k=3
```

---

## 🛠️ How to Create a New Callback

1. Create your python class in `src/utils/evaluation_callbacks.py` inheriting from `lightning.Callback`:
   ```python
   from lightning import Callback

   class MyCustomLogger(Callback):
       def on_train_epoch_end(self, trainer, pl_module):
           print("Epoch finished!")
   ```
2. Create a corresponding YAML file in this folder (e.g. `configs/callbacks/my_logger.yaml`):
   ```yaml
   my_logger:
     _target_: src.utils.evaluation_callbacks.MyCustomLogger
   ```
3. Load the callback:
   ```bash
   uv run src/train.py callbacks=[default,my_logger]
   ```
