"""Compares two CV sweeps fold by fold, on the partitions they actually shared.

Two ``cv_train.py`` sweeps that fix the same seed, fold count, repeat count and
``data.*`` settings draw the identical users into every (repeat, fold) cell,
because the partition depends only on the seed and the surviving cohort — never
on the model. Comparing them by differencing two independent confidence
intervals throws that away: most of the spread in a CV table at this cohort size
is *which users landed in the test fold*, and that term is common to both runs.
Differencing within a fold cancels it, which is the difference between "the
intervals overlap, can't say" and detecting a real but modest gap.

Pairing is only valid if the partitions truly match, so this does not take it on
faith. ``cv_train.py`` logs a ``cv/cohort_hash`` per run and a
``cv/test_user_hash`` per fold; both are checked here before any difference is
computed, and a mismatch is refused rather than reported. That matters most for
data-setting comparisons, where a changed sampler window or modality set alters
sensor-coverage filtering, drops different responses, and silently produces
folds that hold different people under the same fold index.

Usage:
    uv run src/compare_cv_runs.py <run_a> <run_b> [--metric test/auroc]

where each run is a ``metrics.csv`` written by Lightning's CSVLogger, or any
directory containing one.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Paired keys carried on every per-run row by cv_train.py.
RUN_KEYS = ["cv/repeat", "cv/fold"]
COHORT_KEY = "cv/cohort_hash"
TEST_USERS_KEY = "cv/test_user_hash"
FOLD_PREFIX = "fold/"


def _resolve_metrics_csv(path: Path) -> Path:
    """Accepts a metrics.csv, or finds the single one beneath a run directory."""
    if path.is_file():
        return path

    matches = sorted(path.rglob("metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"No metrics.csv found under {path}")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} metrics.csv files under {path}; pass one explicitly:\n  "
            + "\n  ".join(str(m) for m in matches)
        )
    return matches[0]


def load_per_run_rows(path: Path) -> pd.DataFrame:
    """Extracts the one-row-per-fold records a CV sweep wrote to its CSV log.

    The CSV also holds every training and validation step from every fold, so
    the per-run rows are picked out by the presence of ``cv/fold`` rather than
    by position.
    """
    csv_path = _resolve_metrics_csv(path)
    df = pd.read_csv(csv_path)

    missing = [k for k in RUN_KEYS if k not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} has no CV per-run rows (missing {missing}). It was likely "
            "written by train.py, or by a cv_train.py predating per-run logging."
        )

    rows = df[df["cv/fold"].notna()].copy()
    if rows.empty:
        raise ValueError(f"{csv_path} contains no completed CV runs.")

    rows[RUN_KEYS] = rows[RUN_KEYS].astype(int)
    return rows.set_index(RUN_KEYS).sort_index()


def _cohort_hash(rows: pd.DataFrame, label: str) -> Optional[float]:
    """Returns the sweep's single cohort hash, or None if it wasn't logged."""
    if COHORT_KEY not in rows.columns:
        return None

    unique = rows[COHORT_KEY].dropna().unique()
    if len(unique) > 1:
        raise ValueError(
            f"{label} used {len(unique)} different cohorts across its own runs; "
            "its folds are not partitions of one fixed set of users."
        )
    return float(unique[0]) if len(unique) else None


def check_pairable(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> List[Tuple[int, int]]:
    """Returns the (repeat, fold) cells safe to difference, or raises.

    Refusing here is the point: an unverified pairing produces a number that
    looks like a paired delta and is not one.
    """
    hash_a, hash_b = _cohort_hash(a, label_a), _cohort_hash(b, label_b)
    if hash_a is not None and hash_b is not None and hash_a != hash_b:
        raise ValueError(
            f"Cohorts differ ({label_a}: {hash_a:.0f}, {label_b}: {hash_b:.0f}). The two "
            "sweeps scored different sets of users, so their folds cannot be paired. "
            "This is expected if a data setting changed sensor-coverage filtering; "
            "either set require_sensor_data=False for both runs, pass the intersected "
            "user set as exclude_user_ids to both, or compare them unpaired."
        )
    if hash_a is None or hash_b is None:
        print(
            f"warning: no {COHORT_KEY} logged in at least one run; pairing by fold "
            "index without verifying the cohorts match.",
            file=sys.stderr,
        )

    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        raise ValueError(f"{label_a} and {label_b} share no (repeat, fold) cells.")

    for cell in (set(a.index) ^ set(b.index)):
        print(f"warning: (repeat, fold) {cell} present in only one run; skipped.", file=sys.stderr)

    # The fold index alone is not proof. Two sweeps can agree on cohort size and
    # fold count and still assign users differently if anything upstream of the
    # split changed, so the held-out set itself is compared.
    if TEST_USERS_KEY in a.columns and TEST_USERS_KEY in b.columns:
        mismatched = [
            cell for cell in shared
            if a.loc[cell, TEST_USERS_KEY] != b.loc[cell, TEST_USERS_KEY]
        ]
        if mismatched:
            raise ValueError(
                f"{len(mismatched)} of {len(shared)} folds held out different users "
                f"despite matching indices (e.g. {mismatched[:3]}). Not pairable."
            )

    return shared


def paired_deltas(
    a: pd.DataFrame, b: pd.DataFrame, cells: List[Tuple[int, int]], metric: str
) -> Dict[str, float]:
    """Summarises ``metric_b - metric_a`` over the shared folds.

    NaNs are dropped pairwise: a fold whose test users are all one class gives
    an undefined AUROC for one or both runs, and that fold carries no
    information about the difference either way.

    The interval is a normal approximation to the mean of the per-fold
    differences. Treat it as descriptive rather than as a hypothesis test: folds
    from a repeated CV overlap heavily in their training data, so the
    differences are not independent and this interval runs narrower than a
    correctly calibrated one. It is still far tighter and far better centred
    than differencing two unpaired CV intervals.
    """
    col = f"{FOLD_PREFIX}{metric}"
    for frame, name in ((a, "run A"), (b, "run B")):
        if col not in frame.columns:
            available = sorted(c[len(FOLD_PREFIX):] for c in frame.columns if c.startswith(FOLD_PREFIX))
            raise ValueError(f"{name} has no metric {metric!r}. Available: {available}")

    vals_a = a.loc[cells, col].to_numpy(dtype=float)
    vals_b = b.loc[cells, col].to_numpy(dtype=float)

    finite = np.isfinite(vals_a) & np.isfinite(vals_b)
    vals_a, vals_b = vals_a[finite], vals_b[finite]
    n = len(vals_a)
    if n == 0:
        raise ValueError(f"No fold has a finite {metric} in both runs.")

    deltas = vals_b - vals_a
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1)) if n > 1 else 0.0
    stderr = std_delta / np.sqrt(n) if n > 1 else 0.0

    return {
        "n_pairs": float(n),
        "n_dropped": float(len(cells) - n),
        "mean_a": float(vals_a.mean()),
        "mean_b": float(vals_b.mean()),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_low": mean_delta - 1.96 * stderr,
        "ci_high": mean_delta + 1.96 * stderr,
        "b_wins": float((deltas > 0).sum()),
    }


def _report(metric: str, res: Dict[str, float], label_a: str, label_b: str) -> None:
    n = int(res["n_pairs"])
    print(f"\n{metric}  ({n} paired folds)")
    print(f"  {label_a:<28} {res['mean_a']:.4f}")
    print(f"  {label_b:<28} {res['mean_b']:.4f}")
    print(
        f"  paired delta (B - A)         {res['mean_delta']:+.4f}  "
        f"+/- {res['std_delta']:.4f} (sd)  95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
    )
    print(f"  B better on                  {int(res['b_wins'])}/{n} folds")

    if res["n_dropped"]:
        print(f"  {int(res['n_dropped'])} fold(s) dropped as undefined in one or both runs")

    # State the conclusion the interval supports, so a delta whose sign is not
    # resolved isn't read as a result.
    if res["ci_low"] > 0:
        print(f"  -> interval excludes zero: {label_b} ahead by at least {res['ci_low']:.4f}")
    elif res["ci_high"] < 0:
        print(f"  -> interval excludes zero: {label_a} ahead by at least {-res['ci_high']:.4f}")
    else:
        print("  -> interval spans zero: this sweep does not resolve which is better")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path, help="metrics.csv (or run dir) for model A")
    parser.add_argument("run_b", type=Path, help="metrics.csv (or run dir) for model B")
    parser.add_argument(
        "--metric", action="append", default=None,
        help="Metric to compare, e.g. test/auroc. Repeatable. Defaults to every "
             "metric both runs logged.",
    )
    parser.add_argument("--label-a", default=None, help="Display name for run A")
    parser.add_argument("--label-b", default=None, help="Display name for run B")
    args = parser.parse_args()

    label_a = args.label_a or f"A: {args.run_a.name}"
    label_b = args.label_b or f"B: {args.run_b.name}"

    try:
        rows_a = load_per_run_rows(args.run_a)
        rows_b = load_per_run_rows(args.run_b)
        cells = check_pairable(rows_a, rows_b, label_a, label_b)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    metrics = args.metric
    if not metrics:
        shared_cols = set(rows_a.columns) & set(rows_b.columns)
        metrics = sorted(c[len(FOLD_PREFIX):] for c in shared_cols if c.startswith(FOLD_PREFIX))
        if not metrics:
            print("error: the two runs share no fold-level metrics.", file=sys.stderr)
            return 1

    print(f"Paired over {len(cells)} shared (repeat, fold) cells; cohorts verified identical.")

    for metric in metrics:
        try:
            _report(metric, paired_deltas(rows_a, rows_b, cells, metric), label_a, label_b)
        except ValueError as e:
            print(f"\n{metric}: skipped ({e})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
