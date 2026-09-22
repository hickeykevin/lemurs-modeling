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

### 2. `flaml.yaml` & Specialized FLAML Models (AutoML Baselines)
Integrates with Microsoft's **FLAML** (Fast and Lightweight AutoML) library. It automatically runs a search over traditional machine learning models to find the best fit, supporting either raw flattened time-series windows or collapsed summary-statistics/catch22 features (see `featurizer/`).
*   **`flaml.yaml` (Multi-Estimator AutoML)**: Sweeps across LightGBM, Random Forest, XGBoost, and Extra Trees.
    ```bash
    uv run python src/train.py model=flaml
    ```
*   **`flaml_xgboost.yaml` (XGBoost)**: Pinned to XGBoost (`estimator_list: ['xgboost']`).
    ```bash
    uv run python src/train.py model=flaml_xgboost
    ```
*   **`flaml_rf.yaml` (Random Forest)**: Pinned to Random Forest (`estimator_list: ['rf']`).
    ```bash
    uv run python src/train.py model=flaml_rf
    ```
*   **`flaml_lr.yaml` (Logistic Regression)**: Pinned to Logistic Regression tuning L1 and L2 penalties (`estimator_list: ['lrl1', 'lrl2']`).
    ```bash
    uv run python src/train.py model=flaml_lr
    ```
*   **`flaml_svm.yaml` (Support Vector Machine)**: Tuned scikit-learn Pipeline combining `StandardScaler` and `SVC` (`estimator_list: ['svm_pipeline']`), searching over `C`, `kernel` (`rbf`, `linear`), and `gamma`.
    ```bash
    uv run python src/train.py model=flaml_svm
    ```
*   **Feature engineering:** Tabular tree configs compose in `model/featurizer=summary_stats` by default (see `configs/model/featurizer/summary_stats.yaml`), which collapses each modality's sampled `[Time, Modality]` window into per-modality summary statistics (mean, median, std, min, max) instead of feeding raw per-bin values. Alternative: `model/featurizer=catch22`.




### 3. `lag.yaml` (Last-Value Baseline)
A naive lag benchmark that predicts that the user's current symptom state is exactly identical to their last completed survey answer.
*   **Use Case:** Standard "clinical baseline" to check if historical behavior patterns are actually more informative than the most recent survey state.
*   **Dependencies:** Requires configuring the data sampler to return historical lags.
*   **Command:**
    ```bash
    uv run src/train.py model=lag data/sampler=lag
    ```

### 5. `chronos_bolt.yaml` (Amazon Chronos-Bolt Foundation Model)
Leverages Amazon's lightweight, patch-based time-series foundation model (`amazon/chronos-bolt-mini` by default). Unlike older models, it is channel-independent, fast, runs on both CPU and GPU, and accepts any sequence length $T$ without artificial 168-hour padding.
*   **Requirements:** Requires `uv sync --extra chronos2`.
*   **Command:**
    ```bash
    uv run python src/train.py model=chronos_bolt
    ```
*   **Override Model Size:**
    ```bash
    uv run python src/train.py model=chronos_bolt model.net.model_id=amazon/chronos-bolt-small
    ```

### 6. `chronos2.yaml` (OpenMHC Chronos-2 Foundation Model)
The Stanford OpenMHC fine-tune of `amazon/chronos-2`. Strictly requires mapping modalities to a 19-channel Apple HealthKit schema and padding to a 168-hour weekly grid.
*   **Requirements:** Requires `uv sync --extra chronos2`.
*   **Command:**
    ```bash
    uv run python src/train.py model=chronos2
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


### 4. Tabular Feature Engineering (`featurizer/`)
* **Path**: `configs/model/featurizer/`
* **Purpose**: Used by `FLAMLHealthModule` (`flaml.yaml`, `flaml_xgboost.yaml`) to convert a sampled `[Time, Modality]` sensor window into a tabular record for tree models, instead of a raw flatten.
* **Example (`featurizer/summary_stats.yaml`)**: Targets `src.data.components.tabular_features.SummaryStatsFeaturizer` and lists which per-modality statistics to compute (`stats`). Defaults to `exclude_last_n_cols: auto` to automatically drop any trailing cyclic time features added by samplers (`total_cols - len(modalities)`).
* **Example (`featurizer/catch22.yaml`)**: Targets `src.data.components.catch22_features.Catch22Featurizer` — collapses each modality's window into the 22 canonical catch22 time-series features (Lubba et al., 2019). Defaults to `exclude_last_n_cols: auto` to avoid running dynamical features over artificial cyclic time waves. Undefined per-feature values (e.g. on constant/all-zero windows) are replaced by `fill_value`.


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
