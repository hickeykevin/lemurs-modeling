"""Tests for src/utils/fold_identity.py, including a drift tripwire against
the cv_train.py copies it duplicates.

The tripwire matters because cv_train.py is deliberately left untouched during
the eval_plan parity window (see src/utils/fold_identity.py's module docstring),
so the two copies can silently diverge. Importing cv_train from a test is fine
-- its module-scope side effects (env vars, rootutils, checkpoint compat) are
harmless here, unlike importing it from another entry point.
"""

import pandas as pd
import pytest

from src.utils.fold_identity import fold_identity, user_hash


class _StubDataset:
    def __init__(self, data_links):
        self.data_links = data_links


class _StubDataModule:
    def __init__(self, master_df=None, data_test=None):
        if master_df is not None:
            self.master_df = master_df
        if data_test is not None:
            self.data_test = data_test


def _master_df(user_ids=(1, 1, 2, 3)):
    return pd.DataFrame({"app_user_id": list(user_ids), "answer": [0] * len(user_ids)})


def test_user_hash_is_order_and_duplicate_invariant():
    """The hash fingerprints a SET of users -- ordering and repeats must not
    change it, or two runs over the same people would look like different
    cohorts at comparison time."""
    assert user_hash([1, 2, 3]) == user_hash([3, 1, 2])
    assert user_hash([1, 1, 2, 3, 3]) == user_hash([1, 2, 3])


def test_user_hash_differs_for_different_user_sets():
    assert user_hash([1, 2, 3]) != user_hash([1, 2, 4])


def test_user_hash_survives_a_float_column_round_trip():
    """Loggers only take scalars, so the hash has to round-trip through a float
    (and a CSV) without silent rounding -- that's why it's 8 hex digits."""
    h = user_hash(range(50))
    assert h == float(int(h))
    assert h < 2 ** 32
    assert float(str(h)) == h


def test_fold_identity_reports_cohort_and_test_fingerprints():
    dm = _StubDataModule(
        master_df=_master_df((1, 1, 2, 3)),
        data_test=_StubDataset(_master_df((2, 3))),
    )
    identity = fold_identity(dm)

    assert identity["cv/cohort_hash"] == user_hash([1, 2, 3])
    assert identity["cv/cohort_n_users"] == 3.0
    assert identity["cv/cohort_n_responses"] == 4.0
    assert identity["cv/test_user_hash"] == user_hash([2, 3])
    assert identity["cv/test_n_users"] == 2.0
    assert identity["cv/test_n_responses"] == 2.0


def test_fold_identity_is_empty_for_a_datamodule_that_never_set_up():
    """A run with train=False and test=False never builds master_df. Callers
    merge this into a log row, so it must contribute no columns rather than
    raising."""
    assert fold_identity(_StubDataModule()) == {}


def test_fold_identity_omits_test_keys_when_there_is_no_test_set():
    dm = _StubDataModule(master_df=_master_df())
    identity = fold_identity(dm)
    assert "cv/cohort_hash" in identity
    assert not any(k.startswith("cv/test_") for k in identity)


@pytest.mark.parametrize("user_ids", [(1, 2, 3), (7,), tuple(range(30)), ("a", "b")])
def test_user_hash_matches_cv_train_copy(user_ids):
    """Drift tripwire: cv_train.py's copy is the reference during the parity
    window. If these disagree, the duplicated code has diverged."""
    from src.cv_train import _user_hash

    assert user_hash(user_ids) == _user_hash(user_ids)


def test_fold_identity_matches_cv_train_copy():
    """Drift tripwire for the full fingerprint dict, not just the hash."""
    from src.cv_train import _fold_identity

    dm = _StubDataModule(
        master_df=_master_df((1, 1, 2, 3, 5)),
        data_test=_StubDataset(_master_df((3, 5))),
    )
    assert fold_identity(dm) == _fold_identity(dm)
