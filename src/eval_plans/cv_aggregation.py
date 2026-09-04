"""Fold-metric aggregation (mean / spread / 95% interval), shared by the
eval_plan runner's grouped-CV plan.
"""


from typing import Any, Dict, List

import numpy as np
from lightning.pytorch.loggers import Logger

from src.utils import RankedLogger


def aggregate_cv_metrics(
    metrics_by_run: Dict[str, List[float]],
    logger: List[Logger],
    log: RankedLogger,
    total_runs: int,
    log_step_offset: int,
) -> Dict[str, Any]:
    """Summarises per-run metrics as mean, spread, and a 95% interval.

    Reporting a bare mean over folds hides the thing that matters most at this
    cohort size: with small number of users and roughly 1/3 (at moment of writing)
    ever positive, which users land in a given test fold moves the result more
    than most modelling choices do. The spread and interval are therefore reported
    as first-class outputs, not diagnostics — a mean AUROC quoted without them
    is not interpretable.

    NaNs are excluded rather than propagated, since a fold whose test users are
    all one class yields an undefined AUROC that should not erase the others.
    """
    agg_metrics: Dict[str, Any] = {}
    logged: Dict[str, float] = {}

    table_rows = []
    for k, v_list in sorted(metrics_by_run.items()):
        values = np.asarray([v for v in v_list if v is not None and not np.isnan(v)], dtype=float)
        n = len(values)
        if n == 0:
            log.warning(f"{k}: no finite values across {total_runs} runs")
            continue

        mean_val = float(values.mean())
        # Sample standard deviation: these runs are a sample of possible splits,
        # not the population of them.
        std_val = float(values.std(ddof=1)) if n > 1 else 0.0
        stderr = std_val / np.sqrt(n) if n > 1 else 0.0
        ci_low, ci_high = mean_val - 1.96 * stderr, mean_val + 1.96 * stderr

        agg_metrics[f"{k}_mean"] = mean_val
        agg_metrics[f"{k}_std"] = std_val
        agg_metrics[f"{k}_ci_low"] = float(ci_low)
        agg_metrics[f"{k}_ci_high"] = float(ci_high)
        agg_metrics[f"{k}_n_runs"] = float(n)

        # Include cv_summary/ prefixed entries in agg_metrics as well
        agg_metrics[f"cv_summary/{k}_mean"] = mean_val
        agg_metrics[f"cv_summary/{k}_std"] = std_val
        agg_metrics[f"cv_summary/{k}_ci_low"] = float(ci_low)
        agg_metrics[f"cv_summary/{k}_ci_high"] = float(ci_high)
        agg_metrics[f"cv_summary/{k}_n_runs"] = float(n)

        logged.update({
            f"cv_summary/{k}_mean": mean_val,
            f"cv_summary/{k}_std": std_val,
            f"cv_summary/{k}_ci_low": float(ci_low),
            f"cv_summary/{k}_ci_high": float(ci_high),
            f"cv_summary/{k}_n_runs": float(n),
        })

        skipped = "" if n == total_runs else f" [{total_runs - n} undefined]"
        table_rows.append([
            k,
            f"{mean_val:.4f} ± {std_val:.4f}",
            f"[{ci_low:.4f}, {ci_high:.4f}]",
            f"{n}{skipped}",
        ])

        log.info(
            f"{k}: {mean_val:.4f} +/- {std_val:.4f} (sd)  "
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  n={n}{skipped}"
        )

    # Print a Rich terminal table summary and log to file
    try:
        from rich.table import Table
        from rich.console import Console

        rich_table = Table(
            title=f"Cross-Validation Summary ({total_runs} runs)",
            show_header=True,
            header_style="bold cyan",
        )
        rich_table.add_column("Metric", style="bold white")
        rich_table.add_column("Mean ± SD", justify="center")
        rich_table.add_column("95% CI", justify="center")
        rich_table.add_column("N Runs", justify="right")

        for row in table_rows:
            rich_table.add_row(*row)

        console = Console(record=True)
        console.print(rich_table)

        table_text = console.export_text()
        log.info(f"\n{table_text}")
    except Exception:
        pass

    # Log to WandB as an interactive table if WandbLogger is active
    try:
        from lightning.pytorch.loggers.wandb import WandbLogger
        import wandb

        wandb_table = wandb.Table(
            columns=["Metric", "Mean ± SD", "95% CI", "N Runs"],
            data=table_rows,
        )
        for l in logger:
            if isinstance(l, WandbLogger) and hasattr(l, "experiment") and l.experiment is not None:
                l.experiment.log({"cv_summary/table": wandb_table})
    except Exception:
        pass

    # A single row after the per-run rows, so the summary sorts last and cannot
    # be mistaken for another fold.
    for l in logger:
        if hasattr(l, "log_metrics"):
            l.log_metrics(logged, step=log_step_offset + total_runs + 1)

    return agg_metrics
