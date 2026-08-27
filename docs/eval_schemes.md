# Evaluation schemes: which datamodule, which config, which knob

This study's core question is *"given a user's own history, can the model
forecast their near-future risk?"* Every datamodule and config in this
document is a different way of measuring that (plus one that measures a
different question entirely — see [`CVHealthDataModule`](#cvhealthdatamodule-a-different-question)
below). This doc exists so that question stays traceable through the class
hierarchy and config layers without re-deriving it from source each time.

## Class hierarchy

```
HealthDataModule                     (base: cohort building, scaling, one split)
├── split_mode="user"                  random user-grouped split
├── split_mode="longitudinal"          one chronological cut per user
│
├── IndexedHealthDataModule           (+ return_index=True on data_val/data_test)
│   │
│   └── WalkForwardHealthDataModule   (+ fold_sizing: repeated per-user splits)
│       ├── fold_sizing="count"         fixed response COUNT per fold (legacy)
│       ├── fold_sizing="pct"           fixed % of user's history, EXPANDING train
│       └── fold_sizing="cyclic"        fixed-width train, SLIDES + wraps
│
└── CVHealthDataModule                (separate question — see below)
```

Three separate axes compose to answer "which rows go in train/val/test, and
does the run produce an index I can pool predictions across":

1. **`split_mode`** (`HealthDataModule`, base class) — how many times the
   forecasting question is sampled per user, and by what cut.
2. **`fold_sizing`** (`WalkForwardHealthDataModule` only) — for schemes that
   sample the question *multiple* times per user, how each fold's
   train/test boundary is sized.
3. **`return_index`** (`IndexedHealthDataModule` / inherited by
   `WalkForwardHealthDataModule`) — whether `data_val`/`data_test` carry
   enough to join predictions back to source rows, which is what lets
   `PredictionCollectorCallback` pool predictions across folds/users into
   one evaluation set with a user-cluster bootstrap CI, instead of relying
   on `ClassificationMetricsCallback`'s per-batch, row-level
   `torchmetrics.BootStrapper` CI (which understates uncertainty — see
   *Why user-cluster, not row-level* below).

### `HealthDataModule` (base) — `split_mode`

| `split_mode` | Rows go where | Samples the question... |
|---|---|---|
| `"user"` | Random split, grouped by `app_user_id` so no user appears on both sides | Never — this measures generalization to a *new* user, not forecasting. Only relevant on `CVHealthDataModule`; see below. |
| `"longitudinal"` | One chronological train/val/test cut *per user* (earliest responses → train, latest → test) | Once per user |

`split_mode="longitudinal"` is the single-split baseline: every user
contributes exactly one test window, at a fixed late-timeline position.

### `IndexedHealthDataModule` — adds `return_index`

Pure addition on top of `HealthDataModule`, no new split logic. Rebuilds
`data_val`/`data_test` (after the base class's normal `setup()`) with
`return_index=True`, reusing the already-fitted scaler/demographics state.
Exists so a single-split (`split_mode="longitudinal"`) run can use
`PredictionCollectorCallback` and get the same pooled, user-cluster BCa
bootstrap CI the CV schemes below get, instead of a row-level CI.

Use this directly (not through `WalkForwardHealthDataModule`) whenever you
want `split_mode="user"` or `"longitudinal"` with pooled-CI evaluation and
no fold repetition.

### `WalkForwardHealthDataModule` — adds `fold_sizing`

Subclasses `IndexedHealthDataModule` (inherits its `setup()` rebuild step
unchanged — see the class docstring in
[`walk_forward_health_datamodule.py`](../src/data/walk_forward_health_datamodule.py)
for why). Overrides `_split_data` to select one precomputed fold
(`current_fold`) out of a list `CohortSplitter` builds per-user. This is
what turns the single-cut longitudinal question into a *statistically
stronger* one: it samples "can the model forecast this user's near future"
multiple times per user, walking forward through their timeline, then
pools every fold's test predictions into one evaluation set (fold index is
a per-user position, not a shared calendar window across users — pooling,
not per-fold averaging, is what makes cross-user comparison valid).

| `fold_sizing` | Train window | Test window | Every user in every fold? |
|---|---|---|---|
| `"count"` | Expands from user's earliest response, fixed absolute response *count* per step | Fixed absolute count | No — short-history users drop out of later folds one by one; legacy, superseded by `"pct"` in this repo's own configs |
| `"pct"` | Expands from user's earliest response | Fixed *fraction* of that user's own total responses | Yes — every eligible user contributes to every fold |
| `"cyclic"` | Fixed *width* (fraction of user's total), slides forward, **wraps** past 100% back to 0% | Fixed fraction, tiles the user's entire history each cycle | Yes |

`"pct"` and `"cyclic"` answer subtly different questions:

- **`"pct"`** (expanding window) never trains on a user's chronological
  future — realistic for "what would a deployed model have known at this
  point." But fold difficulty is confounded with how much history had
  accumulated by that fold (early folds forecast from little history, late
  folds from a lot).
- **`"cyclic"`** (fixed-width, wraparound) gives every fold the same train
  width and the same test-window size, so folds are directly comparable to
  each other — at the cost of realism: a wrapping fold's train set can
  include the user's *chronological future* relative to its test window.

Purging (see below) is applied at every fold boundary for both — including,
as of the fix in this branch, **both edges** of a cyclic fold's wrapped
train window (see [`CohortSplitter.split_walk_forward_cyclic`](../src/data/components/cohort_splitter.py)).

### `CVHealthDataModule` — a different question

Not part of the forecasting-question family above.
`split_mode="user"`-style grouped, stratified, *repeated* k-fold — measures
generalization to a brand-new user with zero prior history (a zero-shot
deployment-day question), not within-user forecasting. Documented in its
own docstring in [`cv_health_datamodule.py`](../src/data/cv_health_datamodule.py);
mentioned here only so it isn't confused with the walk-forward family.

## Purging

Every split boundary in this pipeline is *purged*: rows within
`purge_hours` of a boundary are dropped from the train side, so a
sampler's lookback window (e.g. "last 48h of steps") can never reach across
the boundary and leak test-adjacent information into a training row. Purge
width defaults to the active sampler's own lookback window
(`lookback_hours_from_sampler`) unless `purge_hours` is set explicitly.

- **`split_mode="longitudinal"`**: one boundary (train→val, val→test).
- **`fold_sizing="pct"`**: one boundary per fold (train→test, expanding).
- **`fold_sizing="cyclic"`**: **two** boundaries per wrapping fold — the
  near edge (train's tail, immediately before test's start) and the far
  edge (the wrapped train slice's tail, cyclically adjacent to test's
  *end*). Both are purged as of this branch's fix; previously only the
  near edge was, which leaked real information at the sweep's actual
  parameters (`train_width_pct + step_pct == 1.0` puts the far edge at
  *zero* gap from test's end on every wrapping fold).

## Why user-cluster, not row-level, bootstrap CIs

The outcome label here is a person-level trait (suicide risk), and
positives are concentrated in a handful of users. A row-level bootstrap
(e.g. `torchmetrics.BootStrapper`, what `ClassificationMetricsCallback`
uses) resamples individual survey responses as if they were independent
draws — but one user's dozens of responses aren't independent evidence
about risk; they're repeated observations of the same underlying trait. A
user-cluster bootstrap resamples whole `app_user_id`s with replacement
instead, so the CI reflects "how much would this score change with a
different draw of *people*," the actual source of uncertainty at this
cohort size. `_cluster_bootstrap_ci` in
[`wf_cv_train.py`](../src/wf_cv_train.py) implements this via
`scipy.stats.bootstrap(method="BCa")` over user indices;
`IndexedHealthDataModule` and `PredictionCollectorCallback` exist
specifically to make this possible for single-split runs too, not just CV.

## Config file → scheme map

Only the "live" configs are listed — see git history for superseded
pre-sweep exploratory configs (`walk_forward.yaml`,
`walk_forward_cyclic.yaml`, `walk_forward_cyclic_4fold.yaml`,
`walk_forward_cyclic_4fold_sweep.yaml`, `longitudinal_sweep.yaml`), deleted
once the sweep-derived hyperparameters and 5-fold cyclic were settled on.

| Config | `_target_` | Scheme | Entry script |
|---|---|---|---|
| [`configs/data/longitudinal_sweep_indexed.yaml`](../configs/data/longitudinal_sweep_indexed.yaml) | `IndexedHealthDataModule` | `split_mode="longitudinal"`, single cut | `src/single_split_eval.py` |
| [`configs/data/walk_forward_pct_sweep.yaml`](../configs/data/walk_forward_pct_sweep.yaml) | `WalkForwardHealthDataModule` | `fold_sizing="pct"`, 4 folds, expanding | `src/wf_cv_train.py` |
| [`configs/data/walk_forward_cyclic_5fold_sweep.yaml`](../configs/data/walk_forward_cyclic_5fold_sweep.yaml) | `WalkForwardHealthDataModule` | `fold_sizing="cyclic"`, 5 folds, wraparound | `src/wf_cv_train.py` |

All three share the same data/model hyperparameters, carried from W&B
sweep `umdhgse9` (see the `walk_forward_cyclic_5fold_sweep.yaml` comment
for the full provenance) — the point of holding these fixed is to compare
the three *evaluation schemes* against each other, not confound that
comparison with a hyperparameter difference.

## Entry scripts

- **`src/wf_cv_train.py`** — loops `current_fold` over a
  `WalkForwardHealthDataModule`'s folds, trains+tests each, pools every
  fold's test predictions, computes pooled metrics + user-cluster BCa CIs.
- **`src/single_split_eval.py`** — same pooled-metrics/CI treatment, for a
  single `IndexedHealthDataModule` split (no fold loop). Imports
  `_compute_pooled_metrics`/`_save_pooled_predictions` from
  `wf_cv_train.py` rather than duplicating them.

## Known sharp edge: Hydra output-directory collisions

Hydra's run directory is timestamped to the second
(`${now:%Y-%m-%d}_${now:%H-%M-%S}`). Launching two of the scripts above in
the same wall-clock second makes them resolve to the *same* output
directory, and the second run's `pooled_predictions.csv` silently
overwrites the first's. Not yet fixed in code — stagger launches by at
least a second, or check `.hydra/config.yaml` in the output dir before
trusting a `pooled_predictions.csv` you didn't watch get written.
