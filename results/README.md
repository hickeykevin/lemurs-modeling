# Results and data provenance

| File | What it is |
|---|---|
| `all_sweeps_results.csv` | The paper's runs, one row per run (1,840 runs: 7 modality sets for RQ1, step + distance for RQ2). |
| `row_manifest_2026-10-07.json` | Which database rows the runs read: per table, the id ranges and a content hash (no data). `data.row_manifest` points here by default. |
| `cohort_by_sampler.csv` | For each of the 20 lookback × bin settings (step + distance): the participants, surveys and positive labels in the pooled test predictions. |
| `sweep_ids.json` | The W&B sweeps the rows came from, and the code commit they ran. |

## The results CSV

- **Paper rows.** The RQ2 table and RQ3 heatmap use the 400 rows with
  `modalities_choice == "step_distance"`: 20 pipelines × 20 settings, all with metrics.
- **Identifying a run.** `model_pkg` + `featurizer_pkg` identify the pipeline (`raw` = binned
  window, `none (raw sequences)` = sequence model); `lookback_hours` + `resample_freq_hours`
  identify the setting.
- **Metrics.** `pooled_*` are computed on all folds' test predictions together, with
  participant-level bootstrap CIs (`*_ci_low`/`*_ci_high`). `fold_test_*_mean` average the
  per-fold values. `pooled_n_users`, `pooled_n_predictions` and `pooled_n_positive` give the
  cohort size.
- **Unfilled columns.** `within_person_auroc_mean/median` are empty: they were never logged
  per run.
- **Five RQ1 runs without metrics.** Four `flaml_tabpfn_ft` calorie runs and
  `tirex_ft` 24h/8h distance have no metrics. None of them is a step + distance run.

## Cohort

39 participants have surveys after the exclusion list in `CohortBuilder` (test and
discontinued accounts: 1, 2, 3, 10, 12, 21, 22, 30, 43, 44, 53, 54, 55). Of these, 29 enter
the step + distance evaluation in at least one setting and 27 in every setting.
Participant 6 is missing from the 24 h settings, and participant 9 from 24 h and 48 h with
6 h or 12 h bins. Participants 5, 8, 11, 14,
15, 16, 17, 18, 20 and 34 never do: they lack step/distance data around enough surveys.

Rebuilding the cohort from the live database through the manifest (2026-10-16) reproduced
the participant, survey and positive counts of all 20 settings exactly.
