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

`single`, `user_cv`, and `walk_forward` do **not** set a default data config —
you must pass a matching `data=` or the run either uses the wrong datamodule
or errors at instantiation. `cyclical` is the one exception: it sets `data=`
for you, and passing your own overrides it back to nothing.

| `eval_plan` | `data=` needed? | Datamodule required | Command |
|---|---|---|---|
| `single` | yes, explicit | any | `python src/train.py data=single_split callbacks=pooled_eval test=True` |
| `user_cv` | yes, explicit | `CVHealthDataModule` | `python src/train.py eval_plan=user_cv data=cv_default test=True` |
| `walk_forward` | yes, explicit | `WalkForwardHealthDataModule` | `python src/train.py eval_plan=walk_forward data=walk_forward_expanding` |
| `cyclical` | set automatically (`walk_forward_cyclic`) | `WalkForwardHealthDataModule`, `fold_sizing=cyclic` | `python src/train.py eval_plan=cyclical` |

`cyclical` is a config, not a separate plan class: it instantiates
`WalkForwardPlan` and overrides `data=walk_forward_cyclic`. Passing your own
`data=` alongside `eval_plan=cyclical` defeats that override — use
`eval_plan=walk_forward data=<your cyclic config>` instead.

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
| `cv_default.yaml` | `CVHealthDataModule` |
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

## Known sharp edge: Hydra output-directory collisions

Hydra's run directory is timestamped to the second. Two runs launched in the
same wall-clock second resolve to the same output directory, and the second
run's `pooled_predictions.csv` silently overwrites the first's. Stagger
launches by at least a second.
