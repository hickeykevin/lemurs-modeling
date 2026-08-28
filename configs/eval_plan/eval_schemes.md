# Evaluation schemes

Four `eval_plan` choices, each with its own default `data=` config — every
command below is complete as written.

| `eval_plan` | Question | Datamodule | Aggregation |
|---|---|---|---|
| `single` | One train/val/test split | `IndexedHealthDataModule` | one metric set |
| `user_cv` | Generalizes to a new user? | `CVHealthDataModule` | mean ± sd, 95% CI across folds |
| `walk_forward` | Forecasts a user's own future? (expanding window) | `WalkForwardHealthDataModule` | pooled predictions, one metric set |
| `cyclical` | Same, fixed-width sliding/wrapping window | `WalkForwardHealthDataModule` | pooled predictions, one metric set |

## Examples

```bash
# Single split (default)
python src/train.py

# Single split, with pooled user-cluster bootstrap CIs
python src/train.py callbacks=pooled_eval test=True

# Subject-wise cross-validation
python src/train.py eval_plan=user_cv test=True

# Walk-forward, expanding window
python src/train.py eval_plan=walk_forward

# Walk-forward, cyclic (fixed-width, wraps)
python src/train.py eval_plan=cyclical

# Any plan against a different data config
python src/train.py eval_plan=walk_forward data=walk_forward_cyclic
```

`user_cv`/`walk_forward`/`cyclical` always run a test pass; `single` only
when `test=True`.

## Notes

- `cyclical` is `walk_forward`'s plan pointed at a different data default —
  not a separate code path.
- `split_mode` (single-split only) and `fold_sizing` (walk-forward only) pick
  the split shape; see each datamodule's docstring for the options.
- `src/cv_train.py`/`wf_cv_train.py` still exist unchanged as the numerical-
  parity reference for `eval_plan`.
- Hydra's output dir is timestamped to the second — two runs launched in the
  same second overwrite each other's outputs.
