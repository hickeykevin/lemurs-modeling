# 🧱 Data Pipeline Components (`src/data/components/`)

This directory contains the modular building blocks used by the data pipeline to build cohorts, clean data, aggregate targets, and sample time-series data.

---

## 📁 File Reference

### 1. ⚙️ [cohort_builder.py](cohort_builder.py)
*   **Purpose**: Handles cohort definition, database extraction, and demographic filtering.
*   **Key Classes**:
    *   `CohortBuilder`: Integrates raw sensor data and Daily EMAs. It handles the database connection, runs cleaning preprocessors, applies time filters, clips extreme values at the 99th percentile, and isolates specific device demographics (`os_filter`).

### 2. 🧹 [preprocessing.py](preprocessing.py)
*   **Purpose**: Encapsulates modality-specific cleaning logic.
*   **Key Classes**:
    *   `ModalityPreprocessor` (Interface): Abstract base class for all data-cleaning processors.
    *   `StepPreprocessor`: Modality-specific cleaning for step count data. It filters out duplicate records, drops records with zero duration that report positive steps, and remaps anomalous sources.

### 3. 📊 [health_dataset.py](health_dataset.py)
*   **Purpose**: The PyTorch `Dataset` wrapper that represents the final modeling cohort.
*   **Key Classes**:
    *   `HealthDataset`: Wraps the cohort labels and sensor dataframes. It utilizes the sampler to extract the required input sequences during its precomputation phase (`_precompute()`) and applies the fitted feature scaler vectorially to prevent data leakage.

### 4. 🎛️ [label_aggregators.py](label_aggregators.py)
*   **Purpose**: Converts raw daily EMA question answers into binary targets.
*   **Key Classes**:
    *   `LabelAggregator` (Interface): Standardizes how question scores are parsed.
    *   `SuicideRiskAggregator`, `SocialStressAggregator`, etc.: Concrete classes that specify which survey question IDs to query, and define scoring threshold rules to label a user day as 0 (low risk/stress) or 1 (high risk/stress).

### 5. ⏱️ [samplers.py](samplers.py)
*   **Purpose**: Extracts feature sequences relative to survey timestamps.
*   **Key Classes**:
    *   `TimeSampler` (Interface): Standardizes time-series sampling parameters.
    *   `BlockSampler`: Samples clean blocks of fixed-duration sensor sequences (e.g., hourly intervals) within a specific lookback window (e.g., the 7 days prior to a survey).
    *   `LagSampler`: Samples sequences using variable offsets or lags relative to the reference timestamp.
