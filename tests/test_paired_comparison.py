"""Tests for fold fingerprinting and the paired comparison of two CV sweeps.

The comparison rests on one claim: two sweeps sharing a seed and a cohort hold
out the identical users in every (repeat, fold) cell, so their metrics can be
differenced within a fold. These tests pin that claim, and pin the refusals that
fire when it does not hold — an unverified pairing yields a number that looks
like a paired delta and is not one, which is worse than no comparison at all.
"""

import types

import numpy as np
import pandas as pd
import pytest

from src.compare_cv_runs import check_pairable, load_per_run_rows, paired_deltas
from src.eval_plans.fold_identity import fold_identity as _fold_identity, user_hash as _user_hash



from tests.test_cv_evaluation import _cohort, _dm


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def test_user_hash_depends_on_the_set_not_the_order():
    assert _user_hash(["u1", "u2"]) == _user_hash(["u2", "u1"])
    assert _user_hash(["u1", "u2"]) == _user_hash(["u2", "u1", "u1"])


def test_user_hash_separates_different_cohorts():
    assert _user_hash(["u1", "u2"]) != _user_hash(["u1", "u3"])
    assert _user_hash(["u1", "u2"]) != _user_hash(["u1", "u2", "u3"])


def test_user_hash_survives_a_float_column_exactly():
    """Pairing compares hashes for equality, so rounding would silently break it."""
    h = _user_hash([f"u{i}" for i in range(50)])
    assert h == float(int(h))
    assert h == float(pd.Series([h]).to_numpy()[0])


def test_fold_identity_reports_cohort_and_held_out_users():
    dm = types.SimpleNamespace(
        master_df=pd.DataFrame({"app_user_id": ["u1", "u1", "u2", "u3"]}),
        data_test=types.SimpleNamespace(data_links=pd.DataFrame({"app_user_id": ["u3", "u3"]})),
    )
    ident = _fold_identity(dm)
    assert ident["cv/cohort_n_users"] == 3
    assert ident["cv/cohort_n_responses"] == 4
    assert ident["cv/test_n_users"] == 1
    assert ident["cv/test_user_hash"] == _user_hash(["u3"])


def test_fold_identity_survives_a_datamodule_without_setup():
    """A missing fingerprint must not take down a sweep that would otherwise run."""
    assert _fold_identity(types.SimpleNamespace()) == {}


# ---------------------------------------------------------------------------
# The invariant the whole comparison depends on
# ---------------------------------------------------------------------------

def test_same_seed_and_cohort_give_identical_held_out_users():
    """The partition is a function of seed and cohort only — never of the model.

    This is what makes model A and model B comparable fold by fold.
    """
    df = _cohort()
    for repeat in range(3):
        for fold in range(5):
            a = _dm(current_repeat=repeat, current_fold=fold)._split_data(df)[2]
            b = _dm(current_repeat=repeat, current_fold=fold)._split_data(df)[2]
            assert _user_hash(a["app_user_id"]) == _user_hash(b["app_user_id"])


def test_a_changed_cohort_changes_the_fold_fingerprint():
    """Dropping users reshuffles the folds, which must show up as a mismatch."""
    full = _cohort()
    reduced = full[full["app_user_id"] != 19]

    same_index_differs = False
    for fold in range(5):
        a = _dm(current_fold=fold)._split_data(full)[2]
        b = _dm(current_fold=fold)._split_data(reduced)[2]
        if _user_hash(a["app_user_id"]) != _user_hash(b["app_user_id"]):
            same_index_differs = True
    assert same_index_differs, "a smaller cohort produced identical folds"


# ---------------------------------------------------------------------------
# Reading run logs
# ---------------------------------------------------------------------------

def _write_log(path, deltas, cohort_hash=1234.0, test_hashes=None, users=6):
    """Writes a metrics.csv shaped like Lightning's, training rows included."""
    rows = []
    for i, ((repeat, fold), value) in enumerate(sorted(deltas.items())):
        rows.append({"step": float(i), "train/loss": 0.5})  # noise the loader must skip
        rows.append({
            "step": 1_000_000 + i,
            "cv/repeat": float(repeat),
            "cv/fold": float(fold),
            "cv/run": float(i + 1),
            "cv/cohort_hash": cohort_hash,
            "cv/test_user_hash": (test_hashes or {}).get((repeat, fold), float(1000 + fold)),
            "cv/test_n_users": float(users),
            "fold/test/auroc": value,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pair(tmp_path, a_vals, b_vals, **b_kw):
    a = load_per_run_rows(_write_log(tmp_path / "a.csv", a_vals))
    b = load_per_run_rows(_write_log(tmp_path / "b.csv", b_vals, **b_kw))
    return a, b


def test_loader_keeps_only_per_run_rows(tmp_path):
    rows = load_per_run_rows(_write_log(tmp_path / "m.csv", {(0, f): 0.6 for f in range(5)}))
    assert len(rows) == 5
    assert "fold/test/auroc" in rows.columns


def test_loader_accepts_a_directory(tmp_path):
    (tmp_path / "run").mkdir()
    _write_log(tmp_path / "run" / "metrics.csv", {(0, f): 0.6 for f in range(5)})
    assert len(load_per_run_rows(tmp_path / "run")) == 5


def test_loader_rejects_a_log_without_cv_rows(tmp_path):
    path = tmp_path / "plain.csv"
    pd.DataFrame([{"step": 0.0, "train/loss": 0.5}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no CV per-run rows"):
        load_per_run_rows(path)


# ---------------------------------------------------------------------------
# Pairing refuses when the partitions do not match
# ---------------------------------------------------------------------------

def test_matching_sweeps_pair_on_every_cell(tmp_path):
    cells = {(r, f): 0.6 for r in range(2) for f in range(5)}
    a, b = _pair(tmp_path, cells, cells)
    assert len(check_pairable(a, b, "A", "B")) == 10


def test_different_cohorts_are_refused(tmp_path):
    cells = {(0, f): 0.6 for f in range(5)}
    a, b = _pair(tmp_path, cells, cells, cohort_hash=9999.0)
    with pytest.raises(ValueError, match="Cohorts differ"):
        check_pairable(a, b, "A", "B")


def test_same_cohort_but_reshuffled_folds_are_refused(tmp_path):
    """The trap: matching fold indices holding different people."""
    cells = {(0, f): 0.6 for f in range(5)}
    shifted = {(0, f): float(1000 + (f + 1) % 5) for f in range(5)}
    a, b = _pair(tmp_path, cells, cells, test_hashes=shifted)
    with pytest.raises(ValueError, match="held out different users"):
        check_pairable(a, b, "A", "B")


def test_a_sweep_whose_own_cohort_drifted_is_refused(tmp_path):
    path = tmp_path / "drift.csv"
    _write_log(path, {(0, f): 0.6 for f in range(5)})
    rows = pd.read_csv(path)
    rows.loc[rows["cv/fold"] == 3, "cv/cohort_hash"] = 777.0
    rows.to_csv(path, index=False)

    good = load_per_run_rows(_write_log(tmp_path / "good.csv", {(0, f): 0.6 for f in range(5)}))
    with pytest.raises(ValueError, match="different cohorts across its own runs"):
        check_pairable(load_per_run_rows(path), good, "A", "B")


def test_only_overlapping_cells_are_paired(tmp_path):
    a, b = _pair(
        tmp_path,
        {(0, f): 0.6 for f in range(5)},
        {(0, f): 0.6 for f in range(3)},
    )
    assert len(check_pairable(a, b, "A", "B")) == 3


# ---------------------------------------------------------------------------
# The delta itself
# ---------------------------------------------------------------------------

def test_paired_delta_recovers_a_known_offset(tmp_path):
    a_vals = {(0, f): 0.60 + 0.01 * f for f in range(5)}
    b_vals = {k: v + 0.05 for k, v in a_vals.items()}
    a, b = _pair(tmp_path, a_vals, b_vals)
    res = paired_deltas(a, b, check_pairable(a, b, "A", "B"), "test/auroc")

    assert res["mean_delta"] == pytest.approx(0.05)
    assert res["b_wins"] == 5
    # A constant offset has no spread, so the interval must exclude zero.
    assert res["ci_low"] > 0


def test_pairing_beats_differencing_unpaired_intervals(tmp_path):
    """The reason to pair: fold difficulty is shared noise, and it cancels.

    Fold-to-fold spread here dwarfs the 0.02 gap, so unpaired intervals overlap
    while the paired interval resolves the sign.
    """
    rng = np.random.RandomState(0)
    fold_difficulty = {(0, f): 0.5 + 0.15 * rng.randn() for f in range(10)}
    a_vals = dict(fold_difficulty)
    b_vals = {k: v + 0.02 for k, v in fold_difficulty.items()}

    a, b = _pair(tmp_path, a_vals, b_vals)
    res = paired_deltas(a, b, check_pairable(a, b, "A", "B"), "test/auroc")

    unpaired_a = np.array(list(a_vals.values()))
    unpaired_b = np.array(list(b_vals.values()))
    a_hi = unpaired_a.mean() + 1.96 * unpaired_a.std(ddof=1) / np.sqrt(len(unpaired_a))
    b_lo = unpaired_b.mean() - 1.96 * unpaired_b.std(ddof=1) / np.sqrt(len(unpaired_b))

    assert b_lo < a_hi, "unpaired intervals were expected to overlap"
    assert res["ci_low"] > 0, "paired interval should still resolve the sign"


def test_folds_undefined_in_either_run_are_dropped_pairwise(tmp_path):
    a_vals = {(0, f): 0.6 for f in range(5)}
    b_vals = {(0, f): 0.7 for f in range(5)}
    b_vals[(0, 2)] = float("nan")
    a, b = _pair(tmp_path, a_vals, b_vals)
    res = paired_deltas(a, b, check_pairable(a, b, "A", "B"), "test/auroc")

    assert res["n_pairs"] == 4
    assert res["n_dropped"] == 1
    assert res["mean_delta"] == pytest.approx(0.1)


def test_unknown_metric_names_what_is_available(tmp_path):
    cells = {(0, f): 0.6 for f in range(5)}
    a, b = _pair(tmp_path, cells, cells)
    with pytest.raises(ValueError, match="test/auroc"):
        paired_deltas(a, b, check_pairable(a, b, "A", "B"), "test/nonexistent")
