# Data Sampler Configurations

This directory contains configuration files for various **Time Sampling** strategies. A sampler determines how raw health metrics (like steps or calories) are sliced and aggregated relative to a survey response timestamp.

## Available Samplers

### 1. `lag.yaml` (The "Lag-1" Baseline)
This is a special non-metric sampler. It ignores sensor data and instead looks up the user's **previous survey answer**.
- **Use case:** Establishing a baseline for how much "yesterday's mood" predicts "today's mood."
- **Command:** `python src/train.py data/sampler=lag model=lag`

### 2. `block.yaml` (Behavioral Time Blocks)
Divides the day into 4 behavioral segments: Sleep (00-08), Morning (08-12), Afternoon (12-17), and Evening (17-24).
- **Use case:** Capturing behavior in intuitive segments rather than hourly grids.
- **Example Usage:** `python src/train.py data/sampler=block`

### 3. `rolling_hour.yaml`
Uses a fixed lookback window (e.g., last 24 hours) and resamples data into hourly bins.
- **Example Usage:** `python src/train.py data/sampler=rolling_hour`

### 4. `offset.yaml`
Samples data based on fixed offsets from the **midnight** of the survey day.
- **Example Usage:** `python src/train.py data/sampler=offset`

### 5. `daily.yaml`
Aggregates data into a single vector representing the entire previous calendar day.
- **Example Usage:** `python src/train.py data/sampler=daily`

---

## Sampler Parameters & Settings Reference

Each sampler has specific hyperparameters you can configure in their `.yaml` files or override via CLI.

### 🕛 Offset & Daily Samplers (`OffsetSampler`)
Both `offset.yaml` and `daily.yaml` instantiate the `OffsetSampler`. The window is calculated relative to **midnight (00:00 AM) of the day the survey was filled out**.

*   `start_offset_hours` (float): The start time of your window, relative to midnight.
*   `end_offset_hours` (float): The end time of your window, relative to midnight.
*   `resample_freq` (string): Time interval for grouping raw steps (e.g. `"1h"`, `"30m"`).

> [!NOTE]
> **Understanding the Offsets for `daily.yaml` vs custom `offset.yaml`:**
> *   **`daily.yaml`** sets `start_offset_hours: -24.0` and `end_offset_hours: 0.0`. This defines a lookback window spanning from midnight yesterday (`-24.0` hours before midnight today) to midnight today (`0.0`). It captures the entire preceding calendar day in hourly bins.
> *   If you want to sample only the **morning hours of the survey day** (e.g., 00:00 to 09:00 AM today), you would set:
>     `start_offset_hours: 0.0` and `end_offset_hours: 9.0`.
> *   If you want to look at the **afternoon and evening of the previous day** (e.g., 12:00 PM yesterday to 12:00 AM today):
>     `start_offset_hours: -12.0` and `end_offset_hours: 0.0`.

---

### ⏱️ Rolling Sampler (`RollingSampler`)
Calculates a dynamic, moving window backwards from the **exact timestamp of the survey response**.

*   `lookback_hours` (float): The total number of hours immediately preceding the survey response to collect (e.g. `24.0` for one day).
*   `resample_freq` (string): Time interval for resampling bins (e.g. `"1h"`).

---

### 📊 Block Sampler (`BlockSampler`)
Divides the time-series into fixed behavioral blocks (Sleep, Morning, Afternoon, Evening) for each lookback day.

*   `lookback_days` (int): The number of full calendar days preceding today's midnight to sample. A value of `1` yields 4 blocks (the 4 periods of yesterday). A value of `7` yields 28 blocks (4 periods per day over the last week).
*   **Behavioral Blocks defined:**
    1.  **Sleep:** 00:00 – 08:00 (weight is damped by `0.05` to prevent over-representing accidental steps)
    2.  **Morning:** 08:00 – 12:00
    3.  **Afternoon:** 12:00 – 17:00
    4.  **Evening:** 17:00 – 24:00

---

### 🔗 Lag Sampler (`LagSampler`)
*   Does not take step or sensor counts.
*   Uses the historical target answer (mood, stress, etc.) from the **user's last completed survey** as features. Used for establishing baseline benchmarks.

---

## Common Modeling Scenarios

Here are some typical research questions and the commands to run them:

### Scenario A: "Does recent activity (last 12h) predict mood?"
Use the `rolling_hour` sampler with a shorter lookback window.
```bash
python src/train.py data/sampler=rolling_hour data.sampler.lookback_hours=12
```

### Scenario B: "How do behavioral patterns over the last week affect risk?"
Use the `block` sampler with a 7-day history to capture weekly cycles.
```bash
python src/train.py data/sampler=block data.sampler.lookback_days=7
```

### Scenario C: "Does sleep quality (midnight to 8 AM) correlate with stress?"
Use the `offset` sampler to target specific hours of the day.
```bash
# Samples 00:00 to 08:00 (offsets are relative to midnight of survey day)
python src/train.py data/sampler=offset data.sampler.start_offset_hours=0 data.sampler.end_offset_hours=8
```

### Scenario D: "Establishing the 'Naive' Benchmark"
Before trusting your LSTM, check if simply guessing the last known state is just as good.
```bash
python src/train.py data/sampler=lag model=lag
```

---

## How to use in Experiments


Hydra allows you to swap samplers from the command line without changing any code.

### Basic Swap
To change the sampler, use the `data/sampler` override:
```bash
python src/train.py data/sampler=rolling_hour
```

### Overriding Parameters
You can also override specific parameters inside a sampler file using the dot notation:
```bash
# Change the lookback period for the block sampler to 3 days
python src/train.py data/sampler=block data.sampler.lookback_days=3

# Change the lookback hours for the rolling sampler
python src/train.py data/sampler=rolling_hour data.sampler.lookback_hours=48
```

## Creating New Samplers
1. Create a new class in `src/data/components/samplers.py` inheriting from `TimeSampler`.
2. Create a corresponding `.yaml` file in this directory.
3. Set the `_target_` to point to your new Python class.
