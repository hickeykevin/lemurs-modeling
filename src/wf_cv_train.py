import os
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

from src.utils.checkpoint_compat import allow_full_checkpoint_loading
from src.utils.prediction_collector import PredictionCollectorCallback
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    register_resolvers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()

# See cv_train.py's CV_LOG_STEP_OFFSET for why: each fold's trainer restarts
# its own step counter from zero, and a step-monotonic logger (W&B,
# TensorBoard) drops a per-run row logged at a colliding step otherwise.
WF_LOG_STEP_OFFSET = 1_000_000

allow_full_checkpoint_loading()


@task_wrapper
def wf_cv_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains and evaluates the model across every walk-forward fold, pooling predictions.

    Unlike ``cv_train.py``'s repeated grouped k-fold (which answers "does this
    generalize to a new user with no history?" and so reports fold metrics
    averaged with a spread), this answers "given a user's own history, can
    the model forecast their near-future risk?" — a question best answered by
    pooling every fold's out-of-fold test predictions into one evaluation set
    rather than averaging per-fold scores. See
    ``WalkForwardHealthDataModule`` and ``CohortSplitter.split_walk_forward``
    for why folds are not directly comparable to each other the way grouped
    CV folds are (fold index is a per-user position, not a shared calendar
    window), which is exactly why pooling — not fold-averaging — is the
    primary aggregation here. A secondary per-fold breakdown is still
    reported (see ``_fold_breakdown_table``), to see whether forecasting
    improves as more history accumulates, but it is a diagnostic, not the
    headline number.

    There is no repeat axis here (unlike ``cv_train.py``'s ``num_repeats``):
    a walk-forward fold sequence is not an independently-reshuffled draw the
    way a grouped k-fold's fold assignment is, so re-running it with a
    different seed would not answer a new question — see
    ``WalkForwardHealthDataModule``'s docstring.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with pooled metrics and a dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Fold count depends on the cohort's actual response-count distribution
    # (see CohortSplitter.split_walk_forward), so it is resolved dynamically,
    # the same way cv_train.py resolves num_folds=-1 (leave-one-user-out).
    # get_num_folds() runs setup() as a side effect (it needs master_df to
    # count folds), so this first datamodule already holds the built cohort
    # afterward -- get_prebuilt_cohort() below reuses it for every fold
    # instead of re-hitting the database each time.
    log.info("Resolving walk-forward fold count from the cohort's response counts...")
    first_datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    if not hasattr(first_datamodule, "get_num_folds"):
        raise ValueError(
            "cfg.data must instantiate a datamodule with get_num_folds() "
            "(e.g. WalkForwardHealthDataModule) for wf_cv_train.py."
        )
    num_folds = first_datamodule.get_num_folds()
    if num_folds == 0:
        raise ValueError(
            "This cohort/config produces zero walk-forward folds -- lower "
            "burn_in_responses/val_responses/step_responses, or check the "
            "cohort's response-count distribution."
        )
    log.info(f"Walk-forward CV: {num_folds} fold(s), no repeats (see docstring).")

    shared_cohort = (
        first_datamodule.get_prebuilt_cohort()
        if hasattr(first_datamodule, "get_prebuilt_cohort") else None
    )

    per_run_rows: List[Dict[str, float]] = []
    pooled_predictions: List[pd.DataFrame] = []
    object_dict_for_logging = {"cfg": cfg, "logger": logger}

    for fold in range(num_folds):
        log.info(f"--- fold {fold + 1}/{num_folds} ---")

        with open_dict(cfg):
            cfg.data.current_fold = fold

        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.data, prebuilt_cohort=shared_cohort
        )
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        collector = PredictionCollectorCallback(fold_index=fold)
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks + [collector], logger=logger
        )

        if fold == 0 and logger:
            log.info("Logging hyperparameters!")
            object_dict_for_logging.update({
                "datamodule": datamodule, "model": model,
                "callbacks": callbacks, "trainer": trainer,
            })
            log_hyperparameters(object_dict_for_logging)

        if cfg.get("train"):
            trainer.fit(model=model, datamodule=datamodule)

        # Unlike cv_train.py, testing here is not gated on cfg.get("test"):
        # pooled test predictions are the entire point of this script, not
        # an optional extra, so skipping it would leave nothing to pool.
        ckpt_path = trainer.checkpoint_callback.best_model_path if getattr(trainer, "checkpoint_callback", None) else ""
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        fold_df = collector.to_dataframe()
        fold_test_df = fold_df[fold_df["stage"] == "test"].copy()
        pooled_predictions.append(fold_test_df)

        run_row: Dict[str, float] = {"cv/fold": float(fold), "cv/n_test_rows": float(len(fold_test_df))}
        for k, v in trainer.callback_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            run_row[f"fold/{k}"] = val
        per_run_rows.append(run_row)

        for l in logger:
            if hasattr(l, "log_metrics"):
                l.log_metrics(run_row, step=WF_LOG_STEP_OFFSET + fold)

    pooled_df = pd.concat(pooled_predictions, ignore_index=True) if pooled_predictions else pd.DataFrame()
    pooled_metrics = _compute_pooled_metrics(pooled_df, log, cfg=cfg)
    fold_breakdown = _fold_breakdown_table(pooled_df, log)

    for l in logger:
        if hasattr(l, "log_metrics"):
            l.log_metrics(pooled_metrics, step=WF_LOG_STEP_OFFSET + num_folds + 1)
        if hasattr(l, "save"):
            l.save()

    # Persist the full pooled prediction table alongside the run's own logs,
    # not just the summary metrics -- this is what a paper's pooled AUROC/CI
    # is computed from, and it should be reproducible/inspectable afterward.
    _save_pooled_predictions(cfg, pooled_df, log)

    object_dict_for_logging["per_run_metrics"] = per_run_rows
    object_dict_for_logging["pooled_predictions"] = pooled_df
    object_dict_for_logging["fold_breakdown"] = fold_breakdown
    return pooled_metrics, object_dict_for_logging


def _classification_metrics_params(cfg: Optional[DictConfig]) -> Dict[str, Any]:
    """Reads the averaging/threshold params ClassificationMetricsCallback was
    configured with for this run (see configs/callbacks/classification_metrics.yaml),
    so the pooled bootstrap metrics use the exact same settings rather than a
    second, possibly-drifted set of defaults. Falls back to that callback's own
    __init__ defaults when cfg is None or the callback isn't in this run's config.
    """
    defaults = {
        "f1_average": "macro",
        "auroc_average": "macro",
        "precision_average": "macro",
        "recall_average": "macro",
        "specificity_average": "macro",
        "sensitivity_at_specificity_average": "macro",
        "min_specificity": 0.9,
    }
    if cfg is None:
        return defaults
    node = cfg.get("callbacks", {}).get("classification_metrics") if hasattr(cfg.get("callbacks", {}), "get") else None
    if node is None:
        return defaults
    return {k: node.get(k, v) for k, v in defaults.items()}


def _pooled_classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, params: Dict[str, Any]
) -> Dict[str, float]:
    """Computes the same 7 metrics ClassificationMetricsCallback logs at test
    time (F1, AUROC, precision, recall, specificity, sensitivity_at_specificity,
    balanced_accuracy — all "task=multiclass, num_classes=2, average=macro" by
    that callback's defaults), using the same torchmetrics classes it uses,
    over one pooled set of predictions rather than per-fold-then-averaged.

    ``probs`` (softmax-normalized per-class probabilities, what
    PredictionCollectorCallback stores) stands in for the raw logits that
    callback passes these metrics — verified to produce identical values,
    since every metric here depends only on rank order / argmax, both of
    which softmax preserves exactly.
    """
    import torch
    from src.utils.evaluation_callbacks import SensitivityAtSpecificityScalar
    from torchmetrics.classification import F1Score, AUROC, Precision, Recall, Specificity, Accuracy

    # np.asarray(..., copy=True) rather than as_tensor(...) directly: probs
    # (a DataFrame .to_numpy() slice) may not be writable, which torch warns
    # about since it can't guarantee tensor writes stay safe otherwise.
    probs_t = torch.from_numpy(np.asarray(probs, dtype=np.float32).copy())
    targets_t = torch.from_numpy(np.asarray(y_true, dtype=np.int64).copy())

    f1 = F1Score(task="multiclass", num_classes=2, average=params["f1_average"])
    auroc = AUROC(task="multiclass", num_classes=2, average=params["auroc_average"])
    precision = Precision(task="multiclass", num_classes=2, average=params["precision_average"])
    recall = Recall(task="multiclass", num_classes=2, average=params["recall_average"])
    specificity = Specificity(task="multiclass", num_classes=2, average=params["specificity_average"])
    sens_at_spec = SensitivityAtSpecificityScalar(
        task="multiclass", num_classes=2,
        min_specificity=params["min_specificity"],
        average=params["sensitivity_at_specificity_average"],
    )
    balanced_accuracy = Accuracy(task="multiclass", num_classes=2, average="macro")

    return {
        "f1": float(f1(probs_t, targets_t)),
        "auroc": float(auroc(probs_t, targets_t)),
        "precision": float(precision(probs_t, targets_t)),
        "recall": float(recall(probs_t, targets_t)),
        "specificity": float(specificity(probs_t, targets_t)),
        "sensitivity_at_specificity": float(sens_at_spec(probs_t, targets_t)),
        "balanced_accuracy": float(balanced_accuracy(probs_t, targets_t)),
    }


def _compute_pooled_metrics(
    pooled_df: pd.DataFrame,
    log: RankedLogger,
    cfg: Optional[DictConfig] = None,
    n_bootstraps: int = 1000,
) -> Dict[str, float]:
    """Pooled classification metrics over every fold's out-of-fold test
    predictions, with a participant-cluster bootstrap CI on each.

    Reports the same 7 metrics ClassificationMetricsCallback logs per-fold at
    test time (F1, AUROC, precision, recall, specificity,
    sensitivity_at_specificity, balanced_accuracy — see
    _pooled_classification_metrics), plus AUPRC (not in that callback, kept
    from this function's original scope since PR-AUC is the more informative
    curve at this cohort's prevalence).

    Resampling by app_user_id, not by row, matches this cohort's actual
    source of variance: positives are concentrated in a handful of
    participants (see CohortSplitter's class docstring), so a row-level
    bootstrap would understate uncertainty by treating each of one user's
    many responses as an independent draw.

    n_bootstraps defaults to 1000 (production quality); lower it in tests
    that only need to check structure/correctness, not CI precision -- 8
    metrics x n_bootstraps BCa resamples is the dominant cost of this
    function (each resample rebuilds every torchmetrics object from
    scratch), so a full 1000 takes tens of seconds.
    """
    metrics: Dict[str, float] = {}
    if pooled_df.empty:
        log.warning("Pooled prediction table is empty; no folds produced test predictions.")
        return metrics

    prob_cols = sorted(c for c in pooled_df.columns if c.startswith("prob_class_"))
    if len(prob_cols) != 2:
        log.warning(
            f"wf_cv_train's pooled metrics currently assume binary classification "
            f"(found {len(prob_cols)} probability columns); skipping metrics."
        )
        return metrics

    y_true = pooled_df["y_true"].to_numpy()
    probs = pooled_df[prob_cols].to_numpy()  # [N, 2], columns already prob_class_0/1 order

    if len(np.unique(y_true)) < 2:
        log.warning("Pooled test predictions contain only one class; metrics undefined.")
        return metrics

    from sklearn.metrics import average_precision_score

    params = _classification_metrics_params(cfg)
    point = _pooled_classification_metrics(y_true, probs, params)
    for name, value in point.items():
        metrics[f"pooled/{name}"] = value
    metrics["pooled/auprc"] = float(average_precision_score(y_true, probs[:, 1]))
    metrics["pooled/n_predictions"] = float(len(pooled_df))
    metrics["pooled/n_users"] = float(pooled_df["app_user_id"].nunique())
    metrics["pooled/n_positive"] = float(y_true.sum())

    ci = _cluster_bootstrap_ci(pooled_df, prob_cols, params, n_bootstraps=n_bootstraps, seed=0, log=log)
    metrics.update(ci)

    log.info(
        f"Pooled AUROC: {metrics['pooled/auroc']:.4f}  "
        f"(95% user-cluster bootstrap CI [{ci['pooled/auroc_ci_low']:.4f}, "
        f"{ci['pooled/auroc_ci_high']:.4f}], n={int(metrics['pooled/n_predictions'])} "
        f"predictions from {int(metrics['pooled/n_users'])} users)"
    )
    log.info(
        f"Pooled AUPRC: {metrics['pooled/auprc']:.4f}  "
        f"(95% user-cluster bootstrap CI [{ci['pooled/auprc_ci_low']:.4f}, "
        f"{ci['pooled/auprc_ci_high']:.4f}])"
    )
    for name in ("f1", "precision", "recall", "specificity", "sensitivity_at_specificity", "balanced_accuracy"):
        log.info(
            f"Pooled {name}: {metrics[f'pooled/{name}']:.4f}  "
            f"(95% user-cluster bootstrap CI [{ci[f'pooled/{name}_ci_low']:.4f}, "
            f"{ci[f'pooled/{name}_ci_high']:.4f}])"
        )
    return metrics


def _cluster_bootstrap_ci(
    pooled_df: pd.DataFrame,
    prob_cols: List[str],
    params: Dict[str, Any],
    n_bootstraps: int,
    seed: int,
    log: RankedLogger,
) -> Dict[str, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI for every pooled metric
    _compute_pooled_metrics reports (AUROC, AUPRC, F1, precision, recall,
    specificity, sensitivity_at_specificity, balanced_accuracy), resampling
    whole users with replacement.

    Resampling by app_user_id, not by row, matches this cohort's actual
    source of variance: positives are concentrated in a handful of
    participants (see CohortSplitter's class docstring), so a row-level
    bootstrap would understate uncertainty by treating each of one user's
    many responses as an independent draw.

    Uses scipy.stats.bootstrap(method="BCa") rather than a plain percentile
    cut. These metrics are bounded in [0, 1] and, at the scores this pipeline
    typically produces, sit close enough to that ceiling that their
    bootstrap distribution is skewed rather than symmetric; with as few as
    ~28-30 independent clusters (users), a naive percentile interval can be
    visibly biased relative to the true sampling distribution. BCa corrects
    for that skew (the "bias-corrected" part) and for the statistic's
    variance changing with its own value (the "accelerated" part), using a
    jackknife (leave-one-user-out) estimate computed internally by scipy —
    still a resampling-based, no-normality-assumed interval, just a
    better-calibrated one than a plain percentile method.

    ``data=(user_idx,)`` is an array of *indices into the unique-user list*,
    one entry per user — this is what scipy resamples (with replacement)
    for every bootstrap replicate and jackknife pass. Each metric's stat
    function expands whichever user indices a given resample drew into
    their pooled rows and computes that one metric, so the actual
    clustering (resample users, not rows) happens inside the statistic
    function while scipy handles the resampling loop, the jackknife
    bias/acceleration correction, and the interval construction. All metrics
    share one resample-user-index sequence per bootstrap replicate (looped
    per-metric, each a fresh scipy.stats.bootstrap call with the same seed),
    so a given replicate's "which users got drawn" is consistent across
    metrics even though scipy computes each metric's interval independently.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import bootstrap, DegenerateDataWarning

    users = pooled_df["app_user_id"].unique()
    n_users = len(users)
    # Pre-split rows by user once, so each replicate/jackknife call is a
    # cheap list lookup + concat rather than a fresh boolean mask over the
    # full pooled_df.
    rows_by_user = {u: pooled_df[pooled_df["app_user_id"] == u] for u in users}

    def _resample_rows(user_idx: np.ndarray) -> Optional[pd.DataFrame]:
        sampled_users = users[user_idx.astype(int)]
        rows = pd.concat([rows_by_user[u] for u in sampled_users], ignore_index=True)
        if len(np.unique(rows["y_true"])) < 2:
            return None  # degenerate resample: only one class present
        return rows

    def _sklearn_stat(user_idx, metric_fn, prob_col):
        rows = _resample_rows(np.asarray(user_idx).reshape(-1))
        if rows is None:
            return float("nan")
        return float(metric_fn(rows["y_true"].to_numpy(), rows[prob_col].to_numpy()))

    def _classification_stat(user_idx, metric_name):
        rows = _resample_rows(np.asarray(user_idx).reshape(-1))
        if rows is None:
            return float("nan")
        values = _pooled_classification_metrics(
            rows["y_true"].to_numpy(), rows[prob_cols].to_numpy(), params
        )
        return values[metric_name]

    stat_fns = {
        "auroc": lambda idx, axis=None: _sklearn_stat(idx, roc_auc_score, prob_cols[1]),
        "auprc": lambda idx, axis=None: _sklearn_stat(idx, average_precision_score, prob_cols[1]),
        "f1": lambda idx, axis=None: _classification_stat(idx, "f1"),
        "precision": lambda idx, axis=None: _classification_stat(idx, "precision"),
        "recall": lambda idx, axis=None: _classification_stat(idx, "recall"),
        "specificity": lambda idx, axis=None: _classification_stat(idx, "specificity"),
        "sensitivity_at_specificity": lambda idx, axis=None: _classification_stat(idx, "sensitivity_at_specificity"),
        "balanced_accuracy": lambda idx, axis=None: _classification_stat(idx, "balanced_accuracy"),
    }

    user_idx_array = np.arange(n_users)
    results = {}
    for name, stat_fn in stat_fns.items():
        # A resample/jackknife sample that happens to draw only one class
        # returns nan above. A handful of nans among thousands of bootstrap
        # values is normal and fine -- but BCa also needs a jackknife
        # (leave-one-user-out) pass, and if a SINGLE user holds all the
        # positives, every leave-one-out sample is degenerate and BCa is
        # not computable at all, not just noisier. scipy detects this and
        # emits DegenerateDataWarning; we surface it as a real warning
        # (not swallow it) and fall back to nan bounds so the caller can
        # see the CI genuinely could not be constructed for this cohort.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=DegenerateDataWarning)
            try:
                res = bootstrap(
                    (user_idx_array,),
                    stat_fn,
                    n_resamples=n_bootstraps,
                    method="BCa",
                    vectorized=False,
                    rng=seed,
                )
                results[f"pooled/{name}_ci_low"] = float(res.confidence_interval.low)
                results[f"pooled/{name}_ci_high"] = float(res.confidence_interval.high)
            except Exception as e:
                log.warning(
                    f"BCa bootstrap CI failed for {name} ({e!r}); reporting nan bounds."
                )
                results[f"pooled/{name}_ci_low"] = float("nan")
                results[f"pooled/{name}_ci_high"] = float("nan")
            for w in caught:
                log.warning(
                    f"BCa bootstrap CI for {name}: {w.message} (cohort likely has too "
                    "few independent positive-holding users for BCa's jackknife "
                    "correction; bounds may be nan or unreliable)."
                )

    results["pooled/n_bootstraps_used"] = float(n_bootstraps)
    return results


def _fold_breakdown_table(pooled_df: pd.DataFrame, log: RankedLogger) -> pd.DataFrame:
    """Per-fold AUROC, as a secondary diagnostic for whether forecasting improves
    with more accumulated history -- not the headline result (see module docstring)."""
    if pooled_df.empty:
        return pd.DataFrame()

    from sklearn.metrics import roc_auc_score

    prob_cols = sorted(c for c in pooled_df.columns if c.startswith("prob_class_"))
    if len(prob_cols) != 2:
        return pd.DataFrame()

    rows = []
    for fold_index, group in pooled_df.groupby("fold_index"):
        y_true = group["y_true"].to_numpy()
        auroc = float(roc_auc_score(y_true, group[prob_cols[1]])) if len(np.unique(y_true)) > 1 else float("nan")
        rows.append({
            "fold_index": fold_index,
            "n_predictions": len(group),
            "n_users": group["app_user_id"].nunique(),
            "n_positive": int(y_true.sum()),
            "auroc": auroc,
        })
    table = pd.DataFrame(rows).sort_values("fold_index")

    try:
        from rich.table import Table
        from rich.console import Console

        rich_table = Table(title="Walk-Forward Per-Fold Breakdown (diagnostic, not headline)", show_header=True, header_style="bold cyan")
        for col in table.columns:
            rich_table.add_column(col, justify="right" if col != "fold_index" else "left")
        for _, row in table.iterrows():
            rich_table.add_row(*[
                f"{v:.4f}" if isinstance(v, float) and col == "auroc" else str(v)
                for col, v in zip(table.columns, row)
            ])
        console = Console(record=True)
        console.print(rich_table)
        log.info(f"\n{console.export_text()}")
    except Exception:
        pass

    return table


def _save_pooled_predictions(cfg: DictConfig, pooled_df: pd.DataFrame, log: RankedLogger) -> None:
    """Writes the pooled prediction table to the run's output directory."""
    output_dir = cfg.paths.output_dir if "paths" in cfg and "output_dir" in cfg.paths else None
    if output_dir is None or pooled_df.empty:
        return
    out_path = os.path.join(output_dir, "pooled_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)
    pooled_df.to_csv(out_path, index=False)
    log.info(f"Saved pooled predictions ({len(pooled_df)} rows) to {out_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="wf_cv_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for walk-forward CV training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with the optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = wf_cv_train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
