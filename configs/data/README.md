# 📊 Data Module Configurations

Data configurations in this directory define how datasets are loaded, preprocessed, aligned, and split before training. These settings are used by Hydra to instantiate the PyTorch Lightning `LightningDataModule` class:
* [health_datamodule.py](file:///Users/khickey/Desktop/lemurs/lemurs-modeling/src/data/health_datamodule.py) (`HealthDataModule`)

---

## 📁 Base Configurations

* [**`default.yaml`**](default.yaml): Configures the longitudinal participant health dataset. Sets default batch sizes, requested database tables (modalities like steps, calories, proximity), demographics mapping, evaluation split strategy, and collapse strategy.

### Core Parameters
* `split_mode` (default: `user`): The strategy for dividing train, validation, and test datasets.
  * `"random"`: Row-level random split.
  * `"user"`: Split by user ID (ensures disjoint populations for out-of-sample/cold-start evaluation).
  * `"longitudinal"`: Temporal split within each user's history (predict future state from past history, enabling personalization).
* `use_demographics` (default: `true`): Whether to fetch and process user-level demographic static features (age, gender identity, LGBT status) as context features.
* `os_filter` (default: `both`): Bins cohort based on platform OS (`both`, `ios`, or `android`) to mitigate sensor drift.

---

## ⚙️ Config Groups (Sub-directories)

The data pipeline configuration is split into modular sub-directories. You can swap options within these groups to customize processing:

### 1. 🕒 Time-Series Samplers
* **Path**: `configs/data/sampler/`
* **Purpose**: Configures how raw longitudinal data sequence windows (e.g. daily, rolling hourly blocks) are sampled and aligned relative to EMA survey responses.
* **Cyclic Time Features**: Samplers like `OffsetSampler`, `RollingSampler`, and `BlockSampler` automatically compute and append cyclical time context features (`sin_hour`, `cos_hour`, `sin_weekday`, `cos_weekday`) at each time step.
* **More Details**: See the [configs/data/sampler/README.md](sampler/README.md) sub-guide.

### 2. 📋 Target Aggregators
* **Path**: `configs/data/aggregator/`
* **Purpose**: Translates multiple raw momentary assessment survey answers into target classification labels (0 or 1) using clinical indicator scoring rules, or aggregates them into continuous target scores for regression tasks.
* **More Details**: See the [configs/data/aggregator/README.md](aggregator/README.md) sub-guide.

### 3. 📈 Feature Scalers
* **Path**: `configs/data/scaler/`
* **Purpose**: Applies feature normalization to time-series sequence tensors to facilitate gradient descent stability.
  * [**`dual.yaml`**](scaler/dual.yaml): Extracts both globally standardized steps (for cohort comparison) and subject-standardized steps (for personal baseline offset), and concatenates them to form a multi-channel sequence without lookahead leakage.
  * [**`subject_standard.yaml`**](scaler/subject_standard.yaml): Standardizes features independently on a per-subject/user level, mapping individuals to their relative deviations.
  * [**`standard.yaml`**](scaler/standard.yaml): Standardizes features globally by removing the population mean and scaling to unit variance.
  * [**`minmax.yaml`**](scaler/minmax.yaml): Scales features globally to a specified range (typically 0 to 1).
  * [**`robust.yaml`**](scaler/robust.yaml): Scales features globally using statistics that are robust to outliers (using interquartile range).
  * [**`none.yaml`**](scaler/none.yaml): Bypasses feature scaling.

---

### 🔄 Daily Survey Collapsing (`collapse_strategy`)
When a user submits multiple survey responses on the same day, the data module collapses them to ensure a single representative target label per calendar day (except for `"none"`). Yes-no question styles are always aggregated using a conservative `max` strategy. All other question styles are collapsed using the configured `collapse_strategy`:
* **`mean`** (default): Takes the average value across responses.
* **`max`**: Takes the maximum value across responses.
* **`min`**: Takes the minimum value across responses.
* **`first`**: Takes the earliest response value of the day.
* **`last`**: Takes the latest response value of the day.
* **`none`**: Bypasses collapsing entirely, treating each survey submission as a separate, independent sample.

---

## 💡 CLI Override Examples

To swap individual config groups:
```bash
# Swap to the rolling hour sampler
uv run src/train.py data/sampler=rolling_hour

# Swap to outlier-robust scaling
uv run src/train.py data/scaler=robust

# Swap to the Dual Scaler (global + subject scaling)
uv run src/train.py data/scaler=dual model.net.input_size=6

# Swap clinical target aggregators (Classification)
uv run src/train.py data/aggregator=suicide_risk
```

To override specific properties:
```bash
# Change batch size and user split strategy
uv run src/train.py data.batch_size=16 data.split_mode=user
```
