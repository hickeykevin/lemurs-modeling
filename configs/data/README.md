# 📊 Data Module Configurations

Data configurations in this directory define how datasets are loaded, preprocessed, aligned, and split before training. These settings are used by Hydra to instantiate the PyTorch Lightning `LightningDataModule` class:
* [health_datamodule.py](file:///home/khickey/lemurs-modeling/src/data/health_datamodule.py) (`HealthDataModule`)

---

## 📁 Base Configurations

* [**`health.yaml`**](health.yaml): Configures the longitudinal participant health dataset. Sets options such as default batch sizes, requested database tables (modalities like steps, calories, proximity), demographic filters, and split strategies.

---

## ⚙️ Config Groups (Sub-directories)

The data pipeline configuration is split into modular sub-directories. You can swap options within these groups to customize processing:

### 1. 🕒 Time-Series Samplers
* **Path**: `configs/data/sampler/`
* **Purpose**: Configures how raw longitudinal data sequence windows (e.g. daily, rolling hourly blocks) are sampled and aligned relative to EMA survey responses.
* **More Details**: See the [configs/data/sampler/README.md](sampler/README.md) sub-guide.

### 2. 📋 Target Aggregators
* **Path**: `configs/data/aggregator/`
* **Purpose**: Translates multiple raw momentary assessment survey answers into target classification labels (0 or 1) using clinical indicator scoring rules.
* **More Details**: See the [configs/data/aggregator/README.md](aggregator/README.md) sub-guide.

### 3. 📈 Feature Scalers
* **Path**: `configs/data/scaler/`
* **Purpose**: Applies feature normalization to time-series sequence tensors to facilitate gradient descent stability.
* **Available Options**:
  * [**`standard.yaml`**](scaler/standard.yaml): Standardizes features by removing the mean and scaling to unit variance.
  * [**`minmax.yaml`**](scaler/minmax.yaml): Scales features to a specified range (typically 0 to 1).
  * [**`robust.yaml`**](scaler/robust.yaml): Scales features using statistics that are robust to outliers (using interquartile range).
  * [**`none.yaml`**](scaler/none.yaml): Bypasses feature scaling.

---

## 💡 CLI Override Examples

To swap individual config groups:
```bash
# Swap to the rolling hour sampler
uv run src/train.py data/sampler=rolling_hour

# Swap to outlier-robust scaling
uv run src/train.py data/scaler=robust

# Swap clinical target aggregators
uv run src/train.py data/aggregator=suicide_risk
```

To override specific properties:
```bash
# Change batch size and user split strategy
uv run src/train.py data.batch_size=16 data.split_mode=user
```
