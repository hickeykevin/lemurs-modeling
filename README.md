# LEMURS modeling: RQ2 / RQ3 reproduction

Code to reproduce the model comparison (RQ2 results table) and the lookback × bin-width
analysis (RQ3 heatmap) of *Passive Sensing of Suicide Risk in College Students: How Far
Back Should We Look?*. Each number in that table and heatmap comes from one run of
`src/train.py`: a pipeline (a table row) on one of 20 lookback × bin-width settings, using
step and distance data.

## Setup

```bash
uv sync --extra tabpfn --extra tirex --extra chronos   # extras only needed for the foundation models
cp .env.example .env                                    # then fill in the database credentials
```

Runs read the LEMURS study database (PostgreSQL behind an SSH gateway), so they need study
data access. No study data is stored in this repository.

## Data: which participants and rows

The database keeps growing, so the paper's cohort is pinned by
`results/row_manifest_2026-10-07.json`. It records, for each table the runs read, the row
ids that existed on 2026-10-07 (17:58 UTC) and a hash of their contents; it holds no data.
With it set (the default, `data.row_manifest`), only those rows are read, and the log says
whether they still match:

```
Table 'step': 16139 pinned rows match the manifest
```

If rows were since edited or deleted, the log warns and results can differ. Set
`data.row_manifest=null` to use the current database instead.

Participants: 39 have surveys after excluding test and discontinued accounts
(`CohortBuilder.DEFAULT_EXCLUDED_USER_IDS`); 27–29 of them have step and distance data
around enough surveys to enter the evaluation, depending on the setting.
`results/cohort_by_sampler.csv` lists them per setting, with survey and positive-label counts.

## Fixed protocol

These are the config defaults, so every run uses them:

| | |
|---|---|
| Label | suicide risk: any of items 2/3/5/7 ≥ 2 or any endorsement of 8/12/13 (`configs/data/aggregator/suicide_risk.yaml`) |
| Input | step + distance, summed into bins over the lookback window before each survey, plus hours slept |
| Scaling | dual scaler: global and per-participant standardisation (`configs/data/scaler/dual.yaml`) |
| Evaluation | cyclical walk-forward CV per participant: 5 folds, each testing 20% of a participant's surveys, 120 h purge around each fold boundary; metrics pooled over all folds' predictions |
| Training | inverse-frequency class weights; tabular models tuned by FLAML for ROC AUC (30 s budget; TabPFN 120 s / 600 s); neural models 75 epochs (fine-tuned encoders 15), no early stopping; seed 7 |

## Running

One run, e.g. XGBoost on summary statistics with a 72 h lookback and 6 h bins:

```bash
uv run python src/train.py model=flaml_xgboost model/featurizer=summary_stats \
    data/sampler=sweep_configs/rolling_72h_6h
```

The sampler is `rolling_<lookback>_<bin>` for lookback 24/48/72/96/120 h and bin 4/6/8/12 h.
The model part of the command for each table row:

| Row | Overrides |
|---|---|
| LR / SVM / RF / XGBoost, raw window | `model=flaml_lr` · `flaml_svm` · `flaml_rf` · `flaml_xgboost` |
| … on summary statistics | add `model/featurizer=summary_stats` |
| … on catch22 features | add `model/featurizer=catch22` |
| LSTM / Transformer | `model=lstm` · `model=transformer` |
| TabPFN-3.5 frozen / fine-tuned | `model=flaml_tabpfn` · `model=flaml_tabpfn_ft` |
| TiRex frozen / fine-tuned | `model=tirex trainer=gpu` · `model=tirex_ft trainer=gpu trainer.max_epochs=15` |
| Chronos-Bolt frozen / fine-tuned | `model=chronos_bolt trainer=gpu` · `model=chronos_bolt_ft trainer=gpu trainer.max_epochs=15` |

The whole grid (20 rows × 20 settings = 400 runs), or part of it:

```bash
bash scripts/run_rq2.sh                                   # everything
SAMPLERS="72h_6h 120h_4h" bash scripts/run_rq2.sh xgboost_stats transformer
```

Add `logger=wandb` (or `EXTRA="logger=wandb"` for the script) to log to Weights & Biases.

## Outputs and reference results

Each run writes to `logs/<task_name>/runs/<timestamp>/`: `pooled_predictions.csv` (one row
per test survey across all folds) and the logged metrics, including the `pooled/*` values
the paper reports (balanced accuracy, AUROC, F1, precision, recall, specificity, AUPRC, with
participant-level bootstrap CIs).

`results/all_sweeps_results.csv` has the paper's runs (one row per run). The RQ2 table and
RQ3 heatmap use its `modalities_choice == "step_distance"` rows: the table gives each
pipeline's mean ± SD over the 20 settings, and the heatmap gives the per-setting values.
`results/sweep_ids.json` maps the rows to their W&B sweeps.

Re-runs match the cohort exactly. Metrics can still differ slightly, because FLAML's search
runs on a wall-clock budget (so it depends on hardware) and GPU training is not bit-for-bit
deterministic.

## Layout

```
configs/   Hydra configs; the defaults are the fixed protocol above
src/       data pipeline (cohort, sampling, scaling), models, evaluation, train.py
scripts/   run_rq2.sh
results/   reference results, row manifest, per-setting cohort
tests/     pytest
```

The full research history (other modality sets, ablations, earlier models) is on the
`model/sampler-model-sweep` branch.
