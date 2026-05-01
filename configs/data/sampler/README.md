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
