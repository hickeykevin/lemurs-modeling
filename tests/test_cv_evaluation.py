"""Tests for repeated stratified grouped CV and within-person AUROC."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.components.cohort_splitter import CohortSplitter
from src.data.cv_health_datamodule import CVHealthDataModule
from src.utils.evaluation_callbacks import WithinPersonAUROCCallback


# ---------------------------------------------------------------------------
# Random splitting is gone
# ---------------------------------------------------------------------------

def test_random_split_mode_rejected():
    """Row-level random splitting leaks users and must fail loudly."""
    df = pd.DataFrame({
        "app_user_id": [1, 1, 2, 2],
        "answer": [0, 1, 0, 1],
        "record_timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
    })
    with pytest.raises(ValueError, match="no longer supported"):
        CohortSplitter(split_mode="random").split(df)


def test_random_split_error_names_the_alternative():
    df = pd.DataFrame({
        "app_user_id": [1, 2],
        "answer": [0, 1],
        "record_timestamp": pd.date_range("2026-01-01", periods=2, freq="D"),
    })
    with pytest.raises(ValueError, match="split_mode='user'"):
        CohortSplitter(split_mode="random").split(df)


def test_user_split_still_supported():
    df = pd.DataFrame({
        "app_user_id": np.repeat(np.arange(10), 4),
        "answer": [0, 1] * 20,
        "record_timestamp": pd.date_range("2026-01-01", periods=40, freq="D"),
    })
    tr, va, te = CohortSplitter(split_mode="user", random_state=0).split(df)
    assert not (set(tr["app_user_id"]) & set(te["app_user_id"]))


# ---------------------------------------------------------------------------
# Grouped, stratified folds
# ---------------------------------------------------------------------------

def _cohort(n_users=20, n_per_user=10, positive_users=(0, 1, 2, 3, 4, 5)):
    """Cohort mirroring the real shape: positives concentrated in a few users."""
    rows = []
    for u in range(n_users):
        for i in range(n_per_user):
            positive = u in positive_users and i % 2 == 0
            rows.append({
                "app_user_id": u,
                "answer": int(positive),
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            })
    return pd.DataFrame(rows)


def _dm(**kw):
    """CVHealthDataModule with splitting exercised directly, no DB needed."""
    from lightning.pytorch.utilities.parsing import AttributeDict

    dm = CVHealthDataModule.__new__(CVHealthDataModule)
    defaults = dict(num_folds=5, current_fold=0, num_repeats=20, current_repeat=0,
                    random_state=7, train_val_test_split=(0.7, 0.15, 0.15),
                    high_risk_user_ids=[])
    defaults.update(kw)
    # hparams is a read-only property; Lightning reads it from _hparams.
    dm._hparams = AttributeDict(defaults)
    return dm


def test_folds_keep_users_disjoint():
    df = _cohort()
    for fold in range(5):
        tr, va, te = _dm(current_fold=fold)._split_data(df)
        tr_u, va_u, te_u = (set(d["app_user_id"]) for d in (tr, va, te))
        assert not (tr_u & te_u), f"fold {fold}: train/test overlap"
        assert not (va_u & te_u), f"fold {fold}: val/test overlap"
        assert not (tr_u & va_u), f"fold {fold}: train/val overlap"


def test_every_fold_gets_positive_test_users():
    """Stratification is the point: an all-negative test fold has no AUROC."""
    df = _cohort()
    for fold in range(5):
        _, _, te = _dm(current_fold=fold)._split_data(df)
        assert te["answer"].sum() > 0, f"fold {fold} has no positive test samples"


def test_folds_cover_all_users_exactly_once():
    df = _cohort()
    seen = []
    for fold in range(5):
        _, _, te = _dm(current_fold=fold)._split_data(df)
        seen.extend(te["app_user_id"].unique())
    assert sorted(seen) == sorted(df["app_user_id"].unique())


def test_repeats_produce_different_partitions():
    """Repeats must reshuffle, otherwise they add runs but no information."""
    df = _cohort()
    partitions = set()
    for repeat in range(5):
        _, _, te = _dm(current_repeat=repeat)._split_data(df)
        partitions.add(frozenset(te["app_user_id"].unique()))
    assert len(partitions) > 1


def test_repeats_are_reproducible():
    df = _cohort()
    a = _dm(current_repeat=3)._split_data(df)[2]["app_user_id"].unique()
    b = _dm(current_repeat=3)._split_data(df)[2]["app_user_id"].unique()
    np.testing.assert_array_equal(sorted(a), sorted(b))


def test_falls_back_when_stratum_too_small():
    """Fewer positive users than folds must still yield usable splits."""
    df = _cohort(n_users=20, positive_users=(0, 1))
    tr, va, te = _dm(num_folds=5)._split_data(df)
    assert len(tr) and len(te)
    assert not (set(tr["app_user_id"]) & set(te["app_user_id"]))


def test_handles_null_seed():
    """cv_train.yaml may leave seed unset; splitting must not crash."""
    df = _cohort()
    tr, va, te = _dm(random_state=None)._split_data(df)
    assert len(tr) and len(te)


# ---------------------------------------------------------------------------
# Within-person AUROC
# ---------------------------------------------------------------------------

def test_within_person_auc_perfect_ranking():
    cb = WithinPersonAUROCCallback()
    assert cb._auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_within_person_auc_inverted_ranking():
    cb = WithinPersonAUROCCallback()
    assert cb._auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_within_person_auc_undefined_for_single_class():
    """A participant who is never positive admits no within-person ranking."""
    cb = WithinPersonAUROCCallback()
    assert cb._auc([0.1, 0.4, 0.7], [0, 0, 0]) is None
    assert cb._auc([0.1, 0.4, 0.7], [1, 1, 1]) is None


def test_within_person_auc_handles_ties():
    """All-tied scores are pure chance, which is 0.5, not 0 or 1."""
    cb = WithinPersonAUROCCallback()
    assert cb._auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_positive_probs_from_two_class_logits():
    cb = WithinPersonAUROCCallback()
    probs = cb._positive_probs(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert probs is not None
    assert probs[0] < 0.5 < probs[1]


def test_positive_probs_rejects_multiclass():
    """Within-person AUROC is defined here for the binary task only."""
    cb = WithinPersonAUROCCallback()
    assert cb._positive_probs(torch.zeros(4, 5)) is None


def test_within_person_differs_from_pooled():
    """The metric exists because pooled AUROC can hide within-person failure.

    Two participants, one high-risk and one low-risk. Scores separate the
    people perfectly but are uninformative — in fact inverted — within each
    person. Pooled AUROC looks strong; within-person AUROC does not.
    """
    cb = WithinPersonAUROCCallback(min_samples_per_user=4)

    user_a_scores, user_a_labels = [0.9, 0.8, 0.7, 0.6], [0, 0, 1, 1]
    user_b_scores, user_b_labels = [0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]

    per_user = [cb._auc(user_a_scores, user_a_labels), cb._auc(user_b_scores, user_b_labels)]
    within = float(np.mean(per_user))

    pooled_scores = np.array(user_a_scores + user_b_scores)
    pooled_labels = np.array(user_a_labels + user_b_labels)
    # Pool ranks A's samples above B's; A is the higher-prevalence person.
    pooled = cb._auc(pooled_scores, pooled_labels)

    assert within == 0.0
    assert pooled > within


def test_high_risk_user_ids_strata():
    """Explicit high_risk_user_ids stratify users based on set membership."""
    df = pd.DataFrame({
        "app_user_id": [10006, 10006, 20001, 20001],
        "answer": [0, 0, 1, 1],
    })
    strata = CVHealthDataModule._user_level_strata(df, high_risk_user_ids=[10006])
    assert list(strata) == [1, 1, 0, 0]

