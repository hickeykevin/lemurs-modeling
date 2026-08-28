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
│       ├── fold_sizing="pct"           fixed % of user's history, EXPANDING train
│       └── fold_sizing="cyclic"        fixed-width train, SLIDES + wraps
│
└── CVHealthDataModule                (separate question — see below)
```

Orthogonal to this hierarchy is the **eval plan** (`eval_plan=` on the command
line: `single`, `user_cv`, `walk_forward`, or `cyclical`), which decides how
many times a datamodule is run and how the results combine — see *Entry points*
below. The datamodule decides which rows land in train/val/test for a given
fold; the plan decides which folds exist. A config pairs one of each; see
[*How `eval_plan` and `split_mode` relate*](#how-eval_plan-and-split_mode-relate)
for which axis actually owns which knob.

Three separate axes compose to answer "which rows go in train/val/test, and
does the run produce an index I can pool predictions across":

1. **`split_mode`** (`HealthDataModule`, base class) — how many times the
   forecasting question is sampled per user, and by what cut.
2. **`fold_sizing`** (`WalkForwardHealthDataModule` only) — for schemes that
   sample the question *multiple* times per user, how each fold's
   train/test boundary is sized.
3. **`return_index`** (`IndexedHealthDataModule` / inherited by
   `WalkForwardHealthDataModule`) — whether `data_val`/`data_test` carry
   enough to join predictions back to source rows, which is what a
   pooled-metrics callback (`PooledMetricsCallback` for a single run,
   `wf_cv_train.py`'s own orchestration for CV) needs to pool predictions
   into one evaluation set with a user-cluster bootstrap CI, instead of
   relying on `ClassificationMetricsCallback`'s per-batch, row-level
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
Exists so a single-split (`split_mode="longitudinal"`) run can attach
`PooledMetricsCallback` (see *Entry points* below) and get the same
pooled, user-cluster BCa bootstrap CI the CV schemes below get, instead of
a row-level CI.

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
Grouped, stratified, *repeated* k-fold — measures
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
[`src/utils/pooled_metrics.py`](../src/utils/pooled_metrics.py) implements
this via `scipy.stats.bootstrap(method="BCa")` over user indices;
`IndexedHealthDataModule` and `PooledMetricsCallback` exist specifically to
make this possible for single-split `train.py` runs too, not just CV.

## Config file → scheme map

Only the "live" configs are listed — see git history for superseded
pre-sweep exploratory configs (`walk_forward.yaml`,
`walk_forward_cyclic.yaml`, `walk_forward_cyclic_4fold.yaml`,
`walk_forward_cyclic_4fold_sweep.yaml`, `longitudinal_sweep.yaml`), deleted
once the sweep-derived hyperparameters and 5-fold cyclic were settled on.

Every strategy runs through `src/train.py`, selected by `eval_plan=`:

| Strategy | Command |
|---|---|
| Single split | `python src/train.py data=longitudinal_sweep_indexed callbacks=pooled_eval test=True` |
| Subject-wise CV | `python src/train.py eval_plan=user_cv data=cv_default test=True` |
| Walk-forward, expanding | `python src/train.py eval_plan=walk_forward data=walk_forward_pct_sweep` |
| Walk-forward, cyclic | `python src/train.py eval_plan=cyclical` |

Note the last two share one plan *class*: **cyclic is not a separate strategy**
at the orchestration level, only a different `fold_sizing` on the same
datamodule. `eval_plan=cyclical` is therefore a config that instantiates
`WalkForwardPlan` and overrides `data=walk_forward_cyclic_5fold_sweep` — a
name on the command line, not a new code path. Because that data override *is*
the config, passing your own `data=` alongside `eval_plan=cyclical` defeats it;
use `eval_plan=walk_forward data=<your cyclic config>` for a different pairing.

### How `eval_plan` and `split_mode` relate

They are separate axes, and `eval_plan` does **not** set `split_mode` — the
data config does. What an eval plan controls is how many times the datamodule
is instantiated and what varies between those runs:

| `eval_plan` | Instantiates | Varies per unit | Datamodule it needs |
|---|---|---|---|
| `single` | once | nothing | any — `split_mode` is whatever the data config says |
| `user_cv` | repeats × folds | `data.current_fold`, `data.current_repeat` | `CVHealthDataModule` (its own grouped k-fold; ignores `split_mode`) |
| `walk_forward` | one per fold | `data.current_fold` | `WalkForwardHealthDataModule` (`_split_data` override; `split_mode` unused) |
| `cyclical` | one per fold | `data.current_fold` | same, pinned to `fold_sizing="cyclic"` |

So `split_mode` is only load-bearing for `eval_plan=single`, where
`HealthDataModule._split_data` actually runs: `"longitudinal"` gives the
chronological per-user cut, `"user"` the grouped random one.

The two CV datamodules don't just ignore it — neither **accepts** it, and
neither passes it to the base class either. Both override `_split_data`, and
every read of `hparams.split_mode` lives inside the `HealthDataModule._split_data`
those overrides replace, so the setting could never have reached them.

`CVHealthDataModule`'s user-grouping guarantee comes from `_grouped_split` on
`app_user_id`, not from `split_mode`.

The practical consequence, verified: `data.split_mode=user` on a `cyclical` or
`user_cv` run is not a silent no-op — Hydra rejects it outright, since those
data configs don't declare the key:

```
Could not override 'data.split_mode'.
Key 'split_mode' is not in struct
```

That failure is the useful behaviour, so resist the `+data.split_mode=user`
that Hydra suggests in the same message: it would force the key in, and
instantiation would then fail on a datamodule that takes no such argument. If
you want to vary how folds are cut under a walk-forward plan, the knob is
`fold_sizing` (or `eval_plan=cyclical` vs. `walk_forward`), not `split_mode`.

The three sweep-derived data configs
([`longitudinal_sweep_indexed`](../configs/data/longitudinal_sweep_indexed.yaml),
[`walk_forward_pct_sweep`](../configs/data/walk_forward_pct_sweep.yaml),
[`walk_forward_cyclic_5fold_sweep`](../configs/data/walk_forward_cyclic_5fold_sweep.yaml))
share the same data/model hyperparameters, carried from W&B sweep `umdhgse9`
(see the cyclic config's comment for full provenance) — the point of holding
these fixed is to compare the *evaluation schemes* against each other, not
confound that comparison with a hyperparameter difference.

See git history for superseded pre-sweep exploratory configs
(`walk_forward.yaml`, `walk_forward_cyclic.yaml`,
`walk_forward_cyclic_4fold.yaml`, `walk_forward_cyclic_4fold_sweep.yaml`,
`longitudinal_sweep.yaml`), deleted once the sweep hyperparameters and 5-fold
cyclic were settled on.

## Entry points

**`src/train.py` is the entry point for every strategy.** It instantiates
`cfg.eval_plan` and hands it to one generic runner; it contains no
strategy-specific branching at all.

- **[`src/eval_plans/`](../src/eval_plans/)** — an `EvalPlan` describes a
  strategy as *data*: which units of work exist (`units()`) and how their
  results combine (`aggregate()`). The runner
  ([`runner.py`](../src/eval_plans/runner.py)) executes every plan through the
  same loop — instantiate datamodule/model/callbacks/trainer, fit, test,
  record — so adding a strategy means writing a small `units()`/`aggregate()`
  pair, not a new orchestration script. Read
  [`base.py`](../src/eval_plans/base.py) first; it documents the protocol and
  its **known boundary** (units are fixed before the first one runs, so nested
  CV or adaptive stopping would need a different protocol).
- **`eval_plan=walk_forward`** (and `cyclical`, the same class) pools every
  fold's out-of-fold predictions into
  one evaluation set. It also *requires* `callbacks=walk_forward`
  (`early_stopping.strict: False`, since a fold's val split can end up empty
  after purging), which its plan config pulls in automatically — an explicit
  `callbacks=...` on the command line still wins.
- **`callbacks=pooled_eval`** adds pooled, user-cluster BCa CIs to a
  *single-split* run via `PooledMetricsCallback`
  ([`src/utils/pooled_metrics_callback.py`](../src/utils/pooled_metrics_callback.py)).
  That is a per-trainer concern, independent of the eval plan; it needs
  `test=True` (off by default in `train.yaml`) since its work happens in
  `on_test_end`.
- **`src/utils/pooled_metrics.py`** — the pure metric/bootstrap functions
  shared by the walk-forward plan and `PooledMetricsCallback`, so neither owns
  logic the other imports sideways.

### The older scripts

`src/cv_train.py` and `src/wf_cv_train.py` still exist and still work
unchanged. They are the reference implementations the `eval_plan` path is
verified for numerical parity against, and a few things still depend on them
directly (`src/app.py`'s Cross Validate mode shells out to `cv_train.py`; the
`suicide_risk_cv.yaml` W&B sweep targets it). Retiring them is a separate
decision, deliberately not bundled with introducing the new path.

While both exist, `src/utils/fold_identity.py` and
`src/utils/cv_aggregation.py` are **duplicates** of helpers inside
`cv_train.py` (which cannot be imported from — it runs `rootutils.setup_root()`
and other side effects at module scope). Their module docstrings say so, and
`tests/test_fold_identity.py`/`tests/test_cv_aggregation.py` assert the copies
agree, as a tripwire against silent drift.

## Known sharp edge: Hydra output-directory collisions

Hydra's run directory is timestamped to the second
(`${now:%Y-%m-%d}_${now:%H-%M-%S}`). Launching two of the scripts above in
the same wall-clock second makes them resolve to the *same* output
directory, and the second run's `pooled_predictions.csv` silently
overwrites the first's. Not yet fixed in code — stagger launches by at
least a second, or check `.hydra/config.yaml` in the output dir before
trusting a `pooled_predictions.csv` you didn't watch get written.
