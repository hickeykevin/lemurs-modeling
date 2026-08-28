# Evaluation schemes

## Class hierarchy

```
HealthDataModule                     (base: cohort building, scaling, one split)
├── split_mode="user"                  random user-grouped split
├── split_mode="longitudinal"          one chronological cut per user
│
├── IndexedHealthDataModule           (+ return_index=True on data_val/data_test)
│   │
│   └── WalkForwardHealthDataModule   (+ fold_sizing: repeated per-user splits)
│       ├── fold_sizing="pct"           fixed % of user's history, EXPANDING train
│       └── fold_sizing="cyclic"        fixed-width train, SLIDES + wraps
│
└── CVHealthDataModule                (separate question — see below)
```

`eval_plan=` (command line) and the datamodule are separate axes: the
datamodule decides which rows land in train/val/test for a given fold; the
plan decides how many folds run and how results combine. A config pairs one
of each.

## `split_mode` (`HealthDataModule`)

| `split_mode` | Rows go where |
|---|---|
| `"user"` | Random split, grouped by `app_user_id` — no user on both sides |
| `"longitudinal"` | One chronological train/val/test cut per user (earliest → train, latest → test) |

Only read by `HealthDataModule._split_data`. `CVHealthDataModule` and
`WalkForwardHealthDataModule` both override `_split_data` and take no
`split_mode` argument — setting `data.split_mode=...` under `eval_plan=user_cv`,
`walk_forward`, or `cyclical` is a Hydra error (`Key 'split_mode' is not in
struct`), not a no-op.

## `fold_sizing` (`WalkForwardHealthDataModule`)

| `fold_sizing` | Train window | Test window |
|---|---|---|
| `"pct"` | Expands from user's earliest response | Fixed fraction of that user's own total responses |
| `"cyclic"` | Fixed width (fraction of user's total), slides and wraps past 100% back to 0% | Fixed fraction, tiles the user's entire history each cycle |

Both purge every fold boundary by `purge_hours` (default: derived from the
sampler's lookback window). `"cyclic"` purges **both** edges of a wrapping
fold's train window (near edge against test's start, wrapped far edge
against test's end); `"pct"` has only the one edge.

## `eval_plan` → datamodule → command

Every `eval_plan` config sets its own `data=` default (an `override /data:`
Hydra default, same mechanism each plan uses), so `eval_plan=<name>` alone is
a complete, runnable command. An explicit `data=...` on the command line
always wins over the default — verified for every plan below, including
`cyclical`.

| `eval_plan` | Default `data=` | Datamodule | Bare command |
|---|---|---|---|
| `single` | `single_split` | `IndexedHealthDataModule`, `split_mode=longitudinal` | `python src/train.py` |
| `user_cv` | `user_cv` | `CVHealthDataModule` | `python src/train.py eval_plan=user_cv test=True` |
| `walk_forward` | `walk_forward_expanding` | `WalkForwardHealthDataModule`, `fold_sizing=pct` | `python src/train.py eval_plan=walk_forward` |
| `cyclical` | `walk_forward_cyclic` | `WalkForwardHealthDataModule`, `fold_sizing=cyclic` | `python src/train.py eval_plan=cyclical` |

`cyclical` is a config, not a separate plan class: it instantiates the same
`WalkForwardPlan` as `walk_forward.yaml`, just defaulted to the cyclic data
config instead of the expanding one. `eval_plan=cyclical data=walk_forward_expanding`
runs the expanding scheme through `cyclical`'s config — same result as
`eval_plan=walk_forward` with no `data=` — since only the default differs.

`walk_forward`/`cyclical` also pull in `callbacks=walk_forward`
(`early_stopping.strict: False`, since a fold's val split can be empty after
purging) unless `callbacks=...` is passed explicitly on the command line,
which wins.

`user_cv`/`walk_forward`/`cyclical` always test (`requires_test`); `single`
tests only when `test=True`.

## Aggregation

- **`single`**: one metric set. Add `callbacks=pooled_eval` for a pooled,
  user-cluster BCa bootstrap CI (needs `test=True`; per-trainer, via
  `PooledMetricsCallback`).
- **`user_cv`**: mean ± sd, 95% interval across folds.
- **`walk_forward` / `cyclical`**: every fold's out-of-fold test predictions
  pooled into one set, one metric computed once (fold index is a per-user
  position, not a shared calendar window, so folds are not averaged).
  `src/utils/pooled_metrics.py` holds the shared bootstrap/metric code.

## Config file → data config map

Live configs only:

| Config | `_target_` |
|---|---|
| `default.yaml` | `HealthDataModule` |
| `single_split.yaml` | `IndexedHealthDataModule` |
| `user_cv.yaml` | `CVHealthDataModule` |
| `walk_forward_expanding.yaml` | `WalkForwardHealthDataModule`, `fold_sizing=pct` |
| `walk_forward_cyclic.yaml` | `WalkForwardHealthDataModule`, `fold_sizing=cyclic` |

`single_split`, `walk_forward_expanding`, and `walk_forward_cyclic` share
hyperparameters (W&B sweep `umdhgse9` — see the cyclic config's comment) so
the three schemes are compared without a hyperparameter confound.

## Entry points

`src/train.py` instantiates `cfg.eval_plan` and hands it to
`src/eval_plans/runner.py`, which runs every plan through the same loop
(datamodule → model → callbacks → trainer → fit → test → record). Adding a
strategy means writing a `units()`/`aggregate()` pair (`src/eval_plans/base.py`
has the protocol).

`src/cv_train.py` and `src/wf_cv_train.py` still exist, unchanged, and are
what `eval_plan` is verified against for numerical parity. `src/app.py`'s
Cross Validate mode and the `suicide_risk_cv.yaml` sweep still target
`cv_train.py` directly. `src/utils/fold_identity.py` and
`src/utils/cv_aggregation.py` are deliberate duplicates of helpers inside
`cv_train.py` (which can't be imported without side effects); drift tripwire
tests assert they agree.

`configs/eval.yaml` (`src/eval.py`'s entry config, for reloading a checkpoint)
also defaults `data=single_split`, tracking `train.py`'s own default so a bare
`train.py` → `eval.py` round-trip reloads without a feature-count mismatch.
Pass `data=...` explicitly to `eval.py` when reloading a checkpoint trained
under a different data config.

## Known sharp edge: Hydra output-directory collisions

Hydra's run directory is timestamped to the second. Two runs launched in the
same wall-clock second resolve to the same output directory, and the second
run's `pooled_predictions.csv` silently overwrites the first's. Stagger
launches by at least a second.
