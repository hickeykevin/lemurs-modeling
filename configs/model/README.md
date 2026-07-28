# 🧠 Model Configurations & Baselines

This directory contains configuration files for the architectures and training targets available in this repository.

---

## 🏛️ Available Model Presets

### 1. `default.yaml` (Deep Learning LSTM)
The primary deep learning model for longitudinal sequence learning. It takes time-series health features (e.g. daily step sequences) and processes them sequentially via a Recurrent Neural Network (LSTM).
*   **Architecture:** Dynamically defined inside `model/net/lstm.yaml`.
*   **Configuration Flags & Ablations:**
    *   `use_sequence_data` (default: `true`): Whether to include sequence data. Should be set to `true` unless performing sequence-data ablation.
    *   `class_weights` (default: `null`): An optional list of float weights (e.g. `[1.0, 16.3]`) for each class to compute weighted Cross Entropy Loss, crucial for countering severe class imbalance.
*   **Optimizers:** Dynamically configured using PyTorch optimizers (e.g. Adam).
*   **Command:**
    ```bash
    uv run src/train.py model=default
    ```

### 2. `flaml.yaml` (AutoML Baseline)
Integrates with Microsoft's **FLAML** (Fast and Lightweight AutoML) library. It flattens the input time-series window and automatically runs a search over traditional machine learning models (LightGBM, Random Forest, XGBoost) to find the best fit.
*   **Use Case:** Establishes a highly competitive non-deep-learning baseline.
*   **Command:**
    ```bash
    uv run src/train.py model=flaml
    ```

### 3. `lag.yaml` (Last-Value Baseline)
A naive lag benchmark that predicts that the user's current symptom state is exactly identical to their last completed survey answer.
*   **Use Case:** Standard "clinical baseline" to check if historical behavior patterns are actually more informative than the most recent survey state.
*   **Dependencies:** Requires configuring the data sampler to return historical lags.
*   **Command:**
    ```bash
    uv run src/train.py model=lag data/sampler=lag
    ```

### 4. `majority.yaml` (Majority Class Baseline)
A naive baseline that completely ignores all input features (steps, calories, history) and always predicts the most frequent class observed in the training set.
*   **Use Case:** Verifies whether a model has actually learned patterns or is simply guessing the most common class.
*   **Command:**
    ```bash
    uv run src/train.py model=majority
    ```

---

## ⚙️ Model Config Groups

Deep learning models (like `default.yaml`) are composed of multiple sub-configurations from specific config groups under `configs/model/`. This modular structure lets you mix and match architectures, optimizers, and learning rate schedules:

### 1. Neural Networks (`net/`)
* **Path**: `configs/model/net/`
* **Purpose**: Configures the raw PyTorch neural network layer class (inheriting from `torch.nn.Module`).
* **Example (`net/lstm.yaml`)**: Targets `src.models.components.simple_lstm.SimpleLSTM` and defines architectural arguments like `input_size`, `hidden_size`, `num_layers`, and `dropout` (active dropout applied between LSTM layers).

### 2. Optimizers (`optimizer/`)
* **Path**: `configs/model/optimizer/`
* **Purpose**: Declares the PyTorch optimizer class to use for computing gradient updates.
* **Example (`optimizer/adam.yaml`)**: Targets `torch.optim.Adam` and defines configuration parameters such as the learning rate (`lr`) and `weight_decay`.

### 3. Learning Rate Schedulers (`scheduler/`)
* **Path**: `configs/model/scheduler/`
* **Purpose**: Controls how the optimizer learning rate decays or adjusts throughout training epochs.
* **Example (`scheduler/cosine.yaml`)**: Targets `torch.optim.lr_scheduler.CosineAnnealingLR` to dynamically reduce learning rates following a cosine curve.

---

## ⚡ CLI Examples for Common Scenarios

### Scenario A: Beating the Simple Baseline
Before deploying or optimizing an LSTM, ensure that it can significantly outperform the majority guess.
```bash
uv run src/train.py model=majority
```

### Scenario B: Tuning the Deep Model (LSTM layers, dropout, and user ID dropout)
You can directly override the nested architecture and regularization parameters from your shell:
```bash
# Double the hidden dim size and set a 30% dropout rate
uv run src/train.py model=default model.net.hidden_size=128 model.net.dropout=0.3

# Set a 20% user ID dropout rate for fallback embedding regularization
uv run src/train.py model=default model.user_id_dropout=0.2

# Apply loss class weights to handle highly imbalanced datasets
uv run src/train.py model=default 'model.class_weights=[1.0,16.3]'
```

### Scenario C: Fast AutoML Benchmark Search
By default, FLAML executes a search budget. You can restrict the time (in seconds) the AutoML search is allowed to run:
```bash
# Allow only a 30-second search budget
uv run src/train.py model=flaml model.automl_config.time_budget=30
```

### Scenario D: Swapping Optimizers
We parameterize optimizers in sub-configs. You can swap optimizers while maintaining the core configuration logic:
```bash
# Swaps Adam to SGD optimizer (looks up configs/model/optimizer/sgd.yaml)
uv run src/train.py model=default model/optimizer=sgd
```
