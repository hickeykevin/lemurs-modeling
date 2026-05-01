# Model Configurations

This directory contains configuration files for the various machine learning models and baselines available in this repository. 

## Available Models

### 1. `health.yaml` (The Primary LSTM)
The flagship model for this project. It uses a recurrent neural network (LSTM) to process time-series health metrics.
- **Components:** Uses sub-configs for `net` (the architecture), `optimizer`, and `scheduler`.
- **Command:** `python src/train.py model=health`

### 2. `flaml.yaml` (AutoML Baseline)
Uses the FLAML library to automatically search for the best traditional machine learning model (e.g., LightGBM, Random Forest).
- **Note:** This treats the sampled time-series as a flat feature vector.
- **Command:** `python src/train.py model=flaml`

### 3. `lag.yaml` (Last-Value Baseline)
A simple "Naive" baseline that predicts the current state is the same as the user's previous survey answer.
- **Note:** Requires `data/sampler=lag` to work correctly.
- **Command:** `python src/train.py model=lag data/sampler=lag`

### 4. `majority.yaml` (Majority Class Baseline)
The simplest possible baseline. It ignores all features and always predicts the most frequent class observed in the training set.
- **Command:** `python src/train.py model=majority`

---

## Common Modeling Scenarios

### Scenario A: "Beating the Majority Class"
The first test for any model. If your LSTM cannot beat this, the data is likely too imbalanced or the features are not predictive.
```bash
python src/train.py model=majority
```

### Scenario B: "Deep Learning vs. Traditional ML"
Compare your LSTM against a state-of-the-art Gradient Boosted Tree (via FLAML).
```bash
# Run LSTM
python src/train.py model=health name=lstm_run

# Run AutoML
python src/train.py model=flaml name=automl_run
```

### Scenario C: "Hyperparameter Tuning (LSTM)"
You can override LSTM parameters like hidden layers or dropout directly from the command line.
```bash
# Change hidden size and dropout rate
python src/train.py model=health model.net.hidden_size=128 model.net.dropout=0.5
```

### Scenario D: "Quick AutoML Search"
Run a very fast (10-second) search to get a rough idea of performance.
```bash
python src/train.py model=flaml model.automl_config.time_budget=10
```

---

## How to use in Experiments

### Basic Swap
```bash
python src/train.py model=flaml
```

### Advanced: Nesting Overrides
Many models use sub-configs. You can swap these individually:
```bash
# Keep the health module but swap the optimizer to SGD
python src/train.py model=health model/optimizer=sgd
```
