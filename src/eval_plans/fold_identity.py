"""Cohort/fold fingerprinting, shared by the eval_plan runner.
"""


import hashlib
from typing import Any, Dict

from lightning import LightningDataModule


def user_hash(user_ids: Any) -> float:
    """Hashes a set of user ids to a value that survives a float metric column.

    Loggers only accept scalars, so the fingerprint has to be a number. Eight
    hex digits keeps it under 2^32, which is exactly representable as a float
    and so round-trips through a CSV column without silent rounding.
    """
    key = ",".join(sorted(str(u) for u in set(user_ids)))
    return float(int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16))


def fold_identity(datamodule: LightningDataModule) -> Dict[str, float]:
    """Fingerprints the cohort and the held-out users of the current fold.

    This is what makes a paired comparison between two configurations checkable
    rather than assumed. The partition for a given (repeat, fold) depends only
    on the seed and the surviving cohort, never on the model -- so two runs that
    share a ``cohort_hash`` are splitting the same people, and two runs whose
    folds also share a ``test_user_hash`` held out the identical users and can
    be differenced fold by fold.

    The cohort is *not* invariant to every data setting: anything that changes
    sensor-coverage filtering (a different sampler window, modality set, or
    collapse strategy) drops different responses and so yields a different
    ``cohort_hash``. Logging it means that case surfaces as a mismatch at
    comparison time instead of quietly producing a pairing that isn't one.

    Returns an empty dict when the datamodule has not been set up (no
    ``master_df``), which is the case for a run with both ``train`` and
    ``test`` disabled -- callers merge this into a log row, so an empty result
    simply contributes no columns.
    """
    identity: Dict[str, float] = {}

    master_df = getattr(datamodule, "master_df", None)
    if master_df is not None and "app_user_id" in master_df:
        identity["cv/cohort_hash"] = user_hash(master_df["app_user_id"])
        identity["cv/cohort_n_users"] = float(master_df["app_user_id"].nunique())
        identity["cv/cohort_n_responses"] = float(len(master_df))

    data_test = getattr(datamodule, "data_test", None)
    test_links = getattr(data_test, "data_links", None)
    if test_links is not None and "app_user_id" in test_links:
        identity["cv/test_user_hash"] = user_hash(test_links["app_user_id"])
        identity["cv/test_n_users"] = float(test_links["app_user_id"].nunique())
        identity["cv/test_n_responses"] = float(len(test_links))

    return identity
