# 🕰️ Data Sampler Configurations

A **Sampler** defines the time-window slicing and aggregation logic used to extract passive sensor data (such as step counts or calorie output) relative to the exact timestamp of an EMA survey response.

---

## 🎛️ Available Sampler Presets

You can select a sampler by setting `data/sampler=preset_name` in the CLI:

### 1. `rolling_hour.yaml`
Extracts a sliding window immediately preceding the survey response and groups the raw sensor records into regular hourly bins.
*   **Best for**: RNNs/LSTMs that expect sequential features representing immediate hourly patterns.
*   **Command**: `uv run src/train.py data/sampler=rolling_hour`

### 2. `block.yaml` (Behavioral Time Blocks)
Slices the data into pre-defined behavioral segments for each day: **Sleep** (00:00-08:00), **Morning** (08:00-12:00), **Afternoon** (12:00-17:00), and **Evening** (17:00-24:00).
*   **Best for**: Traditional ML models or LSTMs focusing on coarse day-part rhythms.
*   **Command**: `uv run src/train.py data/sampler=block`

### 3. `daily.yaml`
Aggregates sensor counts into a single vector representing the entire preceding calendar day (midnight-to-midnight).
*   **Command**: `uv run src/train.py data/sampler=daily`

### 4. `offset.yaml`
Defines custom lookback boundaries relative to midnight of the survey day.
*   **Command**: `uv run src/train.py data/sampler=offset`

### 5. `lag.yaml` (Lag-1 Benchmark)
A helper sampler that ignores physical sensor steps entirely and instead fetches the user's historical answers from their *previous* survey response.
*   **Command**: `uv run src/train.py data/sampler=lag model=lag`

---

## 📖 Parameter Settings Reference

### `OffsetSampler` (`offset.yaml`, `daily.yaml`)
Calculates a static window relative to **midnight of the survey completion day**.
*   `start_offset_hours` (float): Start time relative to midnight (e.g. `-24.0` for midnight yesterday).
*   `end_offset_hours` (float): End time relative to midnight (e.g. `0.0` for midnight today).
*   `resample_freq` (string): Time duration for grouping steps (e.g. `"1h"`, `"30m"`).

> [!NOTE]
> *   **`daily.yaml`** defaults: `start_offset_hours: -24.0`, `end_offset_hours: 0.0`. Slices the preceding calendar day.
> *   To capture the morning of the survey day (00:00 to 09:00 AM): Set `start_offset_hours: 0.0` and `end_offset_hours: 9.0`.

---

### `RollingSampler` (`rolling_hour.yaml`)
Slices a dynamic lookback window backward from the **exact survey completion timestamp**.
*   `lookback_hours` (float): Number of hours to look back (e.g., `24.0`).
*   `resample_freq` (string): Slicing interval (e.g., `"1h"`).

---

### `BlockSampler` (`block.yaml`)
Groups steps into behavioral periods.
*   `lookback_days` (int): Number of historical days to pull. (e.g., `7` returns 28 feature blocks: 4 periods per day for a week).

---

## ⚡ CLI Override Scenarios

### Scenario A: Slicing a Shorter/Longer Sequence
To examine step behavior over only the 12 hours preceding a survey:
```bash
uv run src/train.py data/sampler=rolling_hour data.sampler.lookback_hours=12
```

### Scenario B: Sleep Window Slicing
To analyze physical activity specifically during normal sleep hours (midnight to 8:00 AM) relative to survey day:
```bash
uv run src/train.py data/sampler=offset data.sampler.start_offset_hours=0 data.sampler.end_offset_hours=8
```

### Scenario C: Weekly Cycles Slicing
To test if weekly behavioral patterns affect mood predictions, pass a block sampler looking back a full 7 days:
```bash
uv run src/train.py data/sampler=block data.sampler.lookback_days=7
```

---

## 🛠️ How to Create a New Sampler

1. Add your custom sampler class to [src/data/components/samplers.py](file:///home/khickey/lemurs-modeling/src/data/components/samplers.py), inheriting from `TimeSampler`.
2. Add a YAML configuration file to this folder (e.g. `configs/data/sampler/my_sampler.yaml`):
   ```yaml
   _target_: src.data.components.samplers.MyCustomSamplerClass
   param_one: "value"
   ```
3. Load the sampler:
   ```bash
   uv run src/train.py data/sampler=my_sampler
   ```
