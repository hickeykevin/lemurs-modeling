import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch
from src.data.components.cohort_splitter import CohortSplitter, lookback_hours_from_sampler
from src.data.components.demographics_processor import DemographicsProcessor
from src.data.components.prev_prediction_linker import PrevPredictionLinker
from src.data.components.health_dataset import HealthDataset
from src.data.components.samplers import OffsetSampler


@pytest.fixture
def dummy_splitter_data():
    return pd.DataFrame({
        "app_user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 4],
        "record_timestamp": pd.to_datetime([
            "2026-01-01 08:00:00",
            "2026-01-02 08:00:00",
            "2026-01-03 08:00:00",
            "2026-01-04 08:00:00",
            "2026-01-01 08:00:00",
            "2026-01-02 08:00:00",
            "2026-01-03 08:00:00",
            "2026-01-04 08:00:00",
            "2026-01-01 08:00:00",
            "2026-01-01 08:00:00",
        ]),
        "answer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_cohort_splitter_random_removed(dummy_splitter_data):
    """Row-level random splitting leaks users across the split; see CohortSplitter."""
    import pytest

    splitter = CohortSplitter(split_mode="random", train_val_test_split=(0.6, 0.2, 0.2), random_state=42)
    with pytest.raises(ValueError, match="no longer supported"):
        splitter.split(dummy_splitter_data)


def test_cohort_splitter_user(dummy_splitter_data):
    splitter = CohortSplitter(split_mode="user", train_val_test_split=(0.5, 0.25, 0.25), random_state=42)
    train_df, val_df, test_df = splitter.split(dummy_splitter_data)
    
    train_users = set(train_df["app_user_id"])
    val_users = set(val_df["app_user_id"])
    test_users = set(test_df["app_user_id"])

    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)


def test_cohort_splitter_longitudinal(dummy_splitter_data):
    splitter = CohortSplitter(split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25))
    train_df, val_df, test_df = splitter.split(dummy_splitter_data)

    # For user 1 (4 records): train gets 2, val gets 1, test gets 1
    u1_train = train_df[train_df["app_user_id"] == 1]
    u1_val = val_df[val_df["app_user_id"] == 1]
    u1_test = test_df[test_df["app_user_id"] == 1]

    assert len(u1_train) == 2
    assert len(u1_val) == 1
    assert len(u1_test) == 1
    assert u1_train["record_timestamp"].max() < u1_val["record_timestamp"].min()
    assert u1_val["record_timestamp"].max() < u1_test["record_timestamp"].min()


def test_cohort_splitter_longitudinal_purge_drops_boundary_rows():
    """Rows within purge_hours of a boundary are dropped, not reassigned."""
    # One user, 10 responses spaced 6h apart -> 54h total span.
    # Unpurged (0.5/0.25/0.25): train_end=5, val_end=7 -> train=5, val=2, test=3.
    df = pd.DataFrame({
        "app_user_id": [1] * 10,
        "record_timestamp": pd.date_range("2026-01-01", periods=10, freq="6h"),
        "answer": list(range(10)),
    })

    splitter = CohortSplitter(
        split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25), purge_hours=10.0
    )
    train_df, val_df, test_df = splitter.split(df)

    # Every retained row must be at least purge_hours from the next split's
    # first timestamp.
    purge = pd.Timedelta(hours=10)
    val_start = val_df["record_timestamp"].min()
    test_start = test_df["record_timestamp"].min()
    assert (train_df["record_timestamp"] < val_start - purge).all()
    assert (val_df["record_timestamp"] < test_start - purge).all()

    # Purging only removes rows near a boundary -- it never reassigns a row
    # to a different split.
    unpurged_train, unpurged_val, unpurged_test = CohortSplitter(
        split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25)
    ).split(df)
    assert set(train_df["answer"]).issubset(set(unpurged_train["answer"]))
    assert set(val_df["answer"]).issubset(set(unpurged_val["answer"]))
    assert set(test_df["answer"]).issubset(set(unpurged_test["answer"]))

    # Purging strictly shrinks train/val relative to the unpurged split here.
    assert len(train_df) < len(unpurged_train)
    assert len(val_df) < len(unpurged_val)


def test_cohort_splitter_longitudinal_purge_zero_matches_unpurged():
    """purge_hours=0 (the default) reproduces the original unpurged split exactly."""
    df = pd.DataFrame({
        "app_user_id": [1] * 8,
        "record_timestamp": pd.date_range("2026-01-01", periods=8, freq="12h"),
        "answer": list(range(8)),
    })

    default_split = CohortSplitter(split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25))
    explicit_zero_split = CohortSplitter(
        split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25), purge_hours=0.0
    )

    for a, b in zip(default_split.split(df), explicit_zero_split.split(df)):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_cohort_splitter_swap_val_test_relabels_without_changing_the_cut():
    """swap_val_test relabels the two post-train chunks; it doesn't recut or reroute purging.

    Same fixture as test_cohort_splitter_longitudinal_purge_drops_boundary_rows
    (10 responses, 6h apart, 0.5/0.25/0.25, purge_hours=10). Unswapped: middle
    chunk (purged against the last chunk's start) is returned as val, last
    chunk (never purged) is returned as test. Swapped: same two chunks, exact
    same rows, but middle -> test and last -> val -- so the purge asymmetry
    flips with the label: it's now "test" that's purged, and "val" that isn't.
    """
    df = pd.DataFrame({
        "app_user_id": [1] * 10,
        "record_timestamp": pd.date_range("2026-01-01", periods=10, freq="6h"),
        "answer": list(range(10)),
    })

    unswapped_train, unswapped_val, unswapped_test = CohortSplitter(
        split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25), purge_hours=10.0
    ).split(df)
    swapped_train, swapped_val, swapped_test = CohortSplitter(
        split_mode="longitudinal",
        train_val_test_split=(0.5, 0.25, 0.25),
        purge_hours=10.0,
        swap_val_test=True,
    ).split(df)

    # Train is untouched by the swap.
    pd.testing.assert_frame_equal(
        unswapped_train.reset_index(drop=True), swapped_train.reset_index(drop=True)
    )
    # The middle chunk's rows (purged, previously "val") now come back as test.
    pd.testing.assert_frame_equal(
        unswapped_val.reset_index(drop=True), swapped_test.reset_index(drop=True)
    )
    # The last chunk's rows (unpurged, previously "test") now come back as val.
    pd.testing.assert_frame_equal(
        unswapped_test.reset_index(drop=True), swapped_val.reset_index(drop=True)
    )

    # The purge asymmetry travels with the position, not the label: swapped
    # "test" (middle chunk) is purged against swapped "val" (last chunk)'s start.
    purge = pd.Timedelta(hours=10)
    swapped_val_start = swapped_val["record_timestamp"].min()
    assert (swapped_test["record_timestamp"] < swapped_val_start - purge).all()


def test_cohort_splitter_longitudinal_purge_can_empty_a_split():
    """A purge wider than a user's total span can empty out train and val entirely.

    n=6 hourly responses spans only 5h; a 24h purge against val's and test's
    start times is wider than the whole series, so every train/val row falls
    within the purge window and only the unpurged final split (test) survives.
    This is intentional: purging never reassigns a row, it only ever drops
    ones whose sampler window would otherwise reach across a boundary, so an
    overly wide purge relative to a user's cadence can leave them with no
    train or val contribution at all.
    """
    df = pd.DataFrame({
        "app_user_id": [1] * 6,
        "record_timestamp": pd.date_range("2026-01-01", periods=6, freq="1h"),
        "answer": list(range(6)),
    })

    splitter = CohortSplitter(
        split_mode="longitudinal", train_val_test_split=(0.5, 0.25, 0.25), purge_hours=24.0
    )
    train_df, val_df, test_df = splitter.split(df)
    assert train_df.empty
    assert val_df.empty
    assert not test_df.empty


def test_cohort_splitter_walk_forward_basic_shape():
    """Walk-forward produces sequential, expanding-train folds per user.

    12 hourly responses, burn_in=4, step=2, val=2, no purge:
      fold 0: train=[0:4]  val=[4:6]  test=[6:8]
      fold 1: train=[0:8]  val=[8:10] test=[10:12]
      fold 2: would need test_end=14 > 12 -> stops after fold 1.
    Train expands to absorb the previous fold's test rows (fold 1's train
    includes fold 0's test rows) -- once a response is in the past relative
    to a later test window, the model may train on it.
    """
    df = pd.DataFrame({
        "app_user_id": [1] * 12,
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(
        df, burn_in_responses=4, step_responses=2, val_responses=2
    )

    assert [f.fold_index for f in folds] == [0, 1]

    assert list(folds[0].train_df["answer"]) == [0, 1, 2, 3]
    assert list(folds[0].val_df["answer"]) == [4, 5]
    assert list(folds[0].test_df["answer"]) == [6, 7]

    assert list(folds[1].train_df["answer"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(folds[1].val_df["answer"]) == [8, 9]
    assert list(folds[1].test_df["answer"]) == [10, 11]


def test_cohort_splitter_walk_forward_purges_every_fold_boundary():
    """purge_hours is applied at every fold's train/val and val/test boundary, not just once."""
    df = pd.DataFrame({
        "app_user_id": [1] * 12,
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })

    splitter = CohortSplitter(purge_hours=1.5)
    folds = splitter.split_walk_forward(
        df, burn_in_responses=4, step_responses=2, val_responses=2
    )

    for fold in folds:
        if fold.val_df.empty or fold.test_df.empty:
            continue
        val_start = fold.val_df["record_timestamp"].min()
        test_start = fold.test_df["record_timestamp"].min()
        purge = pd.Timedelta(hours=1.5)
        if not fold.train_df.empty:
            assert (fold.train_df["record_timestamp"] < val_start - purge).all()
        assert (fold.val_df["record_timestamp"] < test_start - purge).all()


def test_cohort_splitter_walk_forward_short_user_contributes_no_folds():
    """A user with too few responses to clear burn-in + val + one step is silently excluded, not an error."""
    df = pd.DataFrame({
        "app_user_id": [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(
        df, burn_in_responses=4, step_responses=2, val_responses=2
    )

    assert len(folds) > 0
    for fold in folds:
        assert (fold.train_df["app_user_id"] != 1).all()
        assert (fold.val_df["app_user_id"] != 1).all()
        assert (fold.test_df["app_user_id"] != 1).all()


def test_cohort_splitter_walk_forward_no_response_is_a_test_example_twice():
    """Pooled-metric honesty check: a given response is a test-role example in at most one fold."""
    df = pd.DataFrame({
        "app_user_id": [1] * 20,
        "record_timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
        "answer": list(range(20)),
    })

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(
        df, burn_in_responses=5, step_responses=3, val_responses=2
    )

    seen_test_answers = []
    for fold in folds:
        seen_test_answers.extend(fold.test_df["answer"].tolist())
    assert len(seen_test_answers) == len(set(seen_test_answers))


def test_cohort_splitter_walk_forward_invalid_args_raise():
    df = pd.DataFrame({
        "app_user_id": [1] * 4,
        "record_timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
        "answer": list(range(4)),
    })
    splitter = CohortSplitter()
    with pytest.raises(ValueError, match="burn_in_responses"):
        splitter.split_walk_forward(df, burn_in_responses=0, step_responses=1, val_responses=1)
    with pytest.raises(ValueError, match="step_responses"):
        splitter.split_walk_forward(df, burn_in_responses=1, step_responses=0, val_responses=1)
    with pytest.raises(ValueError, match="val_responses"):
        splitter.split_walk_forward(df, burn_in_responses=1, step_responses=1, val_responses=-1)


def test_cohort_splitter_walk_forward_val_responses_zero_is_the_default_and_valid():
    """val_responses=0 (the default) is valid: no error, and produces no val window at all."""
    df = pd.DataFrame({
        "app_user_id": [1] * 8,
        "record_timestamp": pd.date_range("2026-01-01", periods=8, freq="1h"),
        "answer": list(range(8)),
    })
    splitter = CohortSplitter(purge_hours=0.0)

    folds_explicit = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=0)
    folds_default = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2)

    for folds in (folds_explicit, folds_default):
        assert len(folds) > 0
        for fold in folds:
            assert fold.val_df.empty
            assert len(fold.val_df.columns) == len(df.columns)  # still correctly-columned, just empty


def test_cohort_splitter_walk_forward_no_val_absorbs_those_rows_into_train():
    """With val_responses=0, train runs right up to the purge boundary before test --
    no responses are set aside and unused the way a val window would set them aside."""
    df = pd.DataFrame({
        "app_user_id": [1] * 12,
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=0)

    # fold 0: train=[0:4), test=[4:6) -- train_end starts at burn_in, test right after.
    assert list(folds[0].train_df["answer"]) == [0, 1, 2, 3]
    assert list(folds[0].test_df["answer"]) == [4, 5]
    assert folds[0].val_df.empty

    # fold 1: train expands to [0:6) (absorbing fold 0's test), test=[6:8).
    assert list(folds[1].train_df["answer"]) == [0, 1, 2, 3, 4, 5]
    assert list(folds[1].test_df["answer"]) == [6, 7]


def test_cohort_splitter_walk_forward_no_val_purges_train_directly_against_test():
    """With val_responses=0, purge is applied directly at the train/test boundary."""
    df = pd.DataFrame({
        "app_user_id": [1] * 12,
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })
    splitter = CohortSplitter(purge_hours=1.5)
    folds = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=0)

    purge = pd.Timedelta(hours=1.5)
    for fold in folds:
        if fold.train_df.empty or fold.test_df.empty:
            continue
        test_start = fold.test_df["record_timestamp"].min()
        assert (fold.train_df["record_timestamp"] < test_start - purge).all()


def test_cohort_splitter_walk_forward_no_user_clears_burn_in_returns_empty():
    """burn_in wider than any user's data yields an empty list, not an error."""
    df = pd.DataFrame({
        "app_user_id": [1, 1, 2, 2],
        "record_timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
        "answer": [0, 1, 0, 1],
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(
        df, burn_in_responses=10, step_responses=1, val_responses=1
    )
    assert folds == []


def test_cohort_splitter_walk_forward_users_contribute_different_fold_counts():
    """A longer-history user contributes more folds; shorter users just stop appearing.

    User 1: 20 responses, burn_in=4, step=2, val=2 -> folds at train_end
    4,8,12,16 (4 folds, since test_end<=20 for all of them: 8,12,16,20).
    User 2: 8 responses -> only fold 0 fits (train_end=4, val 4:6, test 6:8);
    fold 1 would need test_end=12 > 8, so user 2 stops after fold 0.
    """
    rows = []
    for i in range(20):
        rows.append({"app_user_id": 1, "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i), "answer": i})
    for i in range(8):
        rows.append({"app_user_id": 2, "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i), "answer": i})
    df = pd.DataFrame(rows)

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=2)

    assert [f.fold_index for f in folds] == [0, 1, 2, 3]

    # Fold 0: both users present.
    assert set(folds[0].test_df["app_user_id"]) == {1, 2}
    # Folds 1-3: user 2 has run out of data, only user 1 remains.
    for fold in folds[1:]:
        assert set(fold.test_df["app_user_id"]) == {1}
        assert set(fold.train_df["app_user_id"]) == {1}


def test_cohort_splitter_walk_forward_folds_ordered_by_index_then_input_order():
    """Folds are returned sorted by fold_index; users within a fold follow df's own order."""
    rows = []
    for uid in [3, 1, 2]:  # deliberately not sorted, to check output isn't silently re-sorted by user id
        for i in range(10):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "answer": i,
            })
    df = pd.DataFrame(rows)

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=2)

    assert [f.fold_index for f in folds] == sorted(f.fold_index for f in folds)
    # Every fold covers all three users (10 responses/user is enough for 2 folds).
    for fold in folds:
        assert set(fold.test_df["app_user_id"]) == {1, 2, 3}


def test_cohort_splitter_walk_forward_purge_can_empty_a_folds_train_or_val():
    """A purge wider than the gap between chunks can empty train/val for a fold, mirroring longitudinal mode."""
    df = pd.DataFrame({
        "app_user_id": [1] * 12,
        "record_timestamp": pd.date_range("2026-01-01", periods=12, freq="1h"),
        "answer": list(range(12)),
    })
    # 10h purge vs. 1h spacing: train and val should both be fully purged in
    # every fold (their rows all fall within 10h of the next chunk's start).
    splitter = CohortSplitter(purge_hours=10.0)
    folds = splitter.split_walk_forward(df, burn_in_responses=4, step_responses=2, val_responses=2)

    assert len(folds) > 0
    for fold in folds:
        assert fold.train_df.empty
        assert fold.val_df.empty
        assert not fold.test_df.empty  # test is never purged against anything after it


def test_cohort_splitter_walk_forward_pct_basic_shape():
    """burn_in=50%, step=15%, no purge: fold cuts scale with each user's own response count.

    User 1: 20 responses. Fold 0: train=[0,50%)=[0,10), test=[50%,65%)=[10,13).
    Fold 1: train=[0,65%)=[0,13), test=[65%,80%)=[13,16). Fold 2: train=[0,80%)=
    [0,16), test=[80%,95%)=[16,19). Fold 3 would need test_pct_end=1.10 > 1.0 -> stops.
    """
    df = pd.DataFrame({
        "app_user_id": [1] * 20,
        "record_timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
        "answer": list(range(20)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)

    assert [f.fold_index for f in folds] == [0, 1, 2]
    assert list(folds[0].train_df["answer"]) == list(range(10))
    assert list(folds[0].test_df["answer"]) == [10, 11, 12]
    assert list(folds[1].train_df["answer"]) == list(range(13))
    assert list(folds[1].test_df["answer"]) == [13, 14, 15]
    assert list(folds[2].train_df["answer"]) == list(range(16))
    assert list(folds[2].test_df["answer"]) == [16, 17, 18]


def test_cohort_splitter_walk_forward_pct_every_user_contributes_to_every_fold():
    """The whole point of the pct-based split: unlike the fixed-count version,
    users with different total response counts still all contribute to the
    same set of folds, since test size scales with each user's own total."""
    rows = []
    for uid, n in [(1, 20), (2, 50), (3, 100)]:
        for i in range(n):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "answer": i,
            })
    df = pd.DataFrame(rows)

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)

    assert len(folds) > 0
    for fold in folds:
        assert set(fold.test_df["app_user_id"]) == {1, 2, 3}


def test_cohort_splitter_walk_forward_pct_test_fraction_is_consistent_across_users():
    """Each fold's test window is ~step_pct of that user's own total, regardless
    of how many total responses they have -- this is what prevents the tiny
    tail folds split_walk_forward (fixed-count) can produce."""
    rows = []
    for uid, n in [(1, 20), (2, 200)]:
        for i in range(n):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "answer": i,
            })
    df = pd.DataFrame(rows)

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)

    fold0 = folds[0]
    n1 = len(fold0.test_df[fold0.test_df["app_user_id"] == 1])
    n2 = len(fold0.test_df[fold0.test_df["app_user_id"] == 2])
    # 15% of 20 ~= 3, 15% of 200 = 30 -- both close to their own 15%, not a
    # shared absolute count the way split_walk_forward's step_responses is.
    assert n1 == 3
    assert n2 == 30


def test_cohort_splitter_walk_forward_pct_val_pct_zero_is_default_and_no_val_window():
    df = pd.DataFrame({
        "app_user_id": [1] * 20,
        "record_timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
        "answer": list(range(20)),
    })
    splitter = CohortSplitter(purge_hours=0.0)

    folds_default = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)
    folds_explicit = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15, val_pct=0.0)

    for folds in (folds_default, folds_explicit):
        for fold in folds:
            assert fold.val_df.empty


def test_cohort_splitter_walk_forward_pct_purges_train_against_test_start():
    df = pd.DataFrame({
        "app_user_id": [1] * 20,
        "record_timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
        "answer": list(range(20)),
    })
    splitter = CohortSplitter(purge_hours=1.5)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)

    purge = pd.Timedelta(hours=1.5)
    for fold in folds:
        if fold.train_df.empty or fold.test_df.empty:
            continue
        test_start = fold.test_df["record_timestamp"].min()
        assert (fold.train_df["record_timestamp"] < test_start - purge).all()


def test_cohort_splitter_walk_forward_pct_no_response_is_a_test_example_twice():
    df = pd.DataFrame({
        "app_user_id": [1] * 40,
        "record_timestamp": pd.date_range("2026-01-01", periods=40, freq="1h"),
        "answer": list(range(40)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.4, step_pct=0.1)

    seen = []
    for fold in folds:
        seen.extend(fold.test_df["answer"].tolist())
    assert len(seen) == len(set(seen))


def test_cohort_splitter_walk_forward_pct_invalid_args_raise():
    df = pd.DataFrame({
        "app_user_id": [1] * 10,
        "record_timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"),
        "answer": list(range(10)),
    })
    splitter = CohortSplitter()
    with pytest.raises(ValueError, match="burn_in_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=0.0, step_pct=0.1)
    with pytest.raises(ValueError, match="burn_in_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=1.0, step_pct=0.1)
    with pytest.raises(ValueError, match="step_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.0)
    with pytest.raises(ValueError, match="step_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=1.0)
    with pytest.raises(ValueError, match="val_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.1, val_pct=-0.1)
    with pytest.raises(ValueError, match="val_pct"):
        splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.1, val_pct=1.0)


def test_cohort_splitter_walk_forward_pct_short_user_produces_no_folds():
    """A user with too few responses for even one fold's test window to have
    rounding-nonzero width is skipped (see the test_end <= val_end guard)."""
    df = pd.DataFrame({
        "app_user_id": [1] * 3,
        "record_timestamp": pd.date_range("2026-01-01", periods=3, freq="1h"),
        "answer": list(range(3)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_pct(df, burn_in_pct=0.5, step_pct=0.15)
    assert folds == []


def test_cohort_splitter_cyclic_basic_shape():
    """train_width=40%, step=15%, no purge: fold 0's test is the user's earliest
    15% and its train is the FIXED-width 40% immediately preceding it, WRAPPING
    from the end of the sequence back to the start (index 30..49 for n=50).
    """
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)

    assert [f.fold_index for f in folds] == list(range(7))  # ceil(1/0.15) = 7

    fold0 = folds[0]
    assert list(fold0.test_df["answer"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert sorted(fold0.train_df["answer"]) == list(range(30, 50))  # wraps

    # A later, non-wrapping fold behaves like an ordinary fixed-width slide.
    fold3 = folds[3]
    assert list(fold3.test_df["answer"]) == [22, 23, 24, 25, 26, 27, 28, 29]
    assert sorted(fold3.train_df["answer"]) == list(range(2, 22))


def test_cohort_splitter_cyclic_covers_every_response_exactly_once_per_cycle():
    """The whole point: test windows tile the user's full response range with
    no gaps and no response tested on twice within one cycle."""
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)

    seen = []
    for fold in folds:
        seen.extend(fold.test_df["answer"].tolist())
    assert sorted(seen) == list(range(50))


def test_cohort_splitter_cyclic_every_fold_train_is_fixed_width():
    """Unlike the expanding-window methods, every fold's train set is the
    SAME size (modulo rounding) -- it slides, it does not grow."""
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)

    sizes = [len(f.train_df) for f in folds]
    assert max(sizes) - min(sizes) <= 1  # rounding-only variation, no growth trend


def test_cohort_splitter_cyclic_every_user_contributes_to_every_fold():
    rows = []
    for uid, n in [(1, 20), (2, 50), (3, 100)]:
        for i in range(n):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "answer": i,
            })
    df = pd.DataFrame(rows)

    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)

    assert len(folds) > 0
    for fold in folds:
        assert set(fold.test_df["app_user_id"]) == {1, 2, 3}


def test_cohort_splitter_cyclic_no_val_window_ever():
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)
    assert len(folds) > 0
    for fold in folds:
        assert fold.val_df.empty


def test_cohort_splitter_cyclic_purges_train_against_test_start():
    """Only the physically-adjacent slice of a (possibly wrapped) train window
    is purged against test's start -- verified here on a wrapping fold, where
    naive purging by raw index position (rather than the slice adjacent to
    test) would purge the wrong end."""
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=1.5)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)

    purge = pd.Timedelta(hours=1.5)
    for fold in folds:
        if fold.train_df.empty or fold.test_df.empty:
            continue
        test_start = fold.test_df["record_timestamp"].min()
        # Every retained train row must be far enough from test's start --
        # true regardless of which side of a wrap it physically sits on,
        # since a wrapped train window's far slice is many hours away either way.
        gaps = (test_start - fold.train_df["record_timestamp"]).abs()
        assert (gaps >= purge).all() or (fold.train_df["record_timestamp"] < test_start - purge).all()


def test_cohort_splitter_cyclic_no_duplicate_test_role_within_a_fold():
    df = pd.DataFrame({
        "app_user_id": [1] * 50,
        "record_timestamp": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "answer": list(range(50)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)
    for fold in folds:
        assert not fold.test_df.duplicated(subset=["app_user_id", "answer"]).any()


def test_cohort_splitter_cyclic_invalid_args_raise():
    df = pd.DataFrame({
        "app_user_id": [1] * 10,
        "record_timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"),
        "answer": list(range(10)),
    })
    splitter = CohortSplitter()
    with pytest.raises(ValueError, match="train_width_pct"):
        splitter.split_walk_forward_cyclic(df, train_width_pct=0.0, step_pct=0.1)
    with pytest.raises(ValueError, match="train_width_pct"):
        splitter.split_walk_forward_cyclic(df, train_width_pct=1.0, step_pct=0.1)
    with pytest.raises(ValueError, match="step_pct"):
        splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.0)
    with pytest.raises(ValueError, match="step_pct"):
        splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=1.0)


def test_cohort_splitter_cyclic_short_user_produces_no_folds():
    """A single-response user has no room for both a train row and a test
    row in any fold, and is excluded."""
    df = pd.DataFrame({
        "app_user_id": [1],
        "record_timestamp": pd.date_range("2026-01-01", periods=1, freq="1h"),
        "answer": [0],
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.4, step_pct=0.15)
    assert folds == []


def test_cohort_splitter_cyclic_train_is_exact_complement_of_test_when_widths_sum_to_one():
    """When train_width_pct + step_pct == 1.0 (e.g. 0.75/0.25 -- ordinary
    4-fold CV with contiguous, wrapped train blocks), every fold's train set
    should be EXACTLY the rest of the user's data not in that fold's test
    set: no gap, no overlap, with no purging (purge_hours=0 here isolates
    this from the purge-driven shrinkage the purge test already covers)."""
    df = pd.DataFrame({
        "app_user_id": [1] * 40,
        "record_timestamp": pd.date_range("2026-01-01", periods=40, freq="1h"),
        "answer": list(range(40)),
    })
    splitter = CohortSplitter(purge_hours=0.0)
    folds = splitter.split_walk_forward_cyclic(df, train_width_pct=0.75, step_pct=0.25)

    assert len(folds) == 4
    all_answers = set(range(40))
    for fold in folds:
        train_answers = set(fold.train_df["answer"])
        test_answers = set(fold.test_df["answer"])
        assert train_answers.isdisjoint(test_answers)
        assert train_answers | test_answers == all_answers


def test_lookback_hours_from_sampler():
    """Derives purge width from window_bounds rather than a type-specific attribute."""

    class FakeSampler:
        def window_bounds(self, ts):
            ts = pd.Timestamp(ts)
            return ts - pd.Timedelta(hours=96), ts

    class NoWindowSampler:
        """Mimics LagSampler: no window_bounds override, reads no sensor data."""
        pass

    assert lookback_hours_from_sampler(FakeSampler()) == 96.0
    assert lookback_hours_from_sampler(NoWindowSampler()) == 0.0


def test_health_dataset_handles_empty_split():
    """An empty split (e.g. every response purged) yields a valid, empty Dataset
    rather than crashing on np.stack([])."""
    empty_links = pd.DataFrame(columns=["app_user_id", "record_timestamp", "answer"])
    step_df = pd.DataFrame(columns=["app_user_id", "start_timestamp", "steps"])

    dataset = HealthDataset(
        linked_data=empty_links,
        modality_dfs={"step": step_df},
        modality_cols={"step": "steps"},
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
    )

    assert len(dataset) == 0


def test_health_dataset_return_index_appends_idx_last():
    """return_index=True appends idx as the tuple's last element without changing anything else.

    Off by default so existing model_step branches (fixed lengths 4/5/6) are
    unaffected; a caller that opts in can join a prediction back to
    dataset.data_links.iloc[idx] to recover app_user_id/record_timestamp.
    """
    links = pd.DataFrame({
        "app_user_id": [1, 1, 2],
        "record_timestamp": pd.to_datetime([
            "2026-01-01 08:00:00", "2026-01-02 08:00:00", "2026-01-01 08:00:00",
        ]),
        "answer": [0, 1, 0],
    })
    step_df = pd.DataFrame({
        "app_user_id": [1, 1, 2],
        "start_timestamp": pd.to_datetime([
            "2026-01-01 06:00:00", "2026-01-02 06:00:00", "2026-01-01 06:00:00",
        ]),
        "steps": [100, 200, 300],
    })
    sampler = OffsetSampler(start_offset_hours=-24, end_offset_hours=0)

    dataset_without = HealthDataset(
        linked_data=links, modality_dfs={"step": step_df},
        modality_cols={"step": "steps"}, sampler=sampler,
    )
    dataset_with = HealthDataset(
        linked_data=links, modality_dfs={"step": step_df},
        modality_cols={"step": "steps"}, sampler=sampler, return_index=True,
    )

    without_len = len(dataset_without[0])
    with_item = dataset_with[0]
    assert len(with_item) == without_len + 1
    assert with_item[-1].dtype == torch.long
    assert with_item[-1].item() == 0

    # idx recovers the correct source row via data_links.
    for i in range(len(dataset_with)):
        item = dataset_with[i]
        idx = item[-1].item()
        assert idx == i
        assert dataset_with.data_links.iloc[idx]["app_user_id"] == links.iloc[i]["app_user_id"]


def test_demographics_processor():
    train_df = pd.DataFrame({"app_user_id": [1, 2]})
    master_df = pd.DataFrame({"app_user_id": [1, 2, 3]})
    demographics_df = pd.DataFrame({
        "app_user_id": [1, 2, 3],
        "gender": ["male", "female", "unknown"],
        "age": [20, 30, 40],
        "lgbt": ["no", "yes", "no"],
    })
    step_df = pd.DataFrame({
        "app_user_id": [1, 2],
        "app_source": ["iPhone 13", "health.connect.androidx"],
    })
    modality_dfs = {"step": step_df}

    processor = DemographicsProcessor(use_demographics=True)
    demo_map, default_demo = processor.fit_transform(
        train_df=train_df,
        demographics_df=demographics_df,
        modality_dfs=modality_dfs,
        master_df=master_df,
    )

    # Features should include: age_scaled, gender_male, gender_female, gender_unknown, lgbt_no, lgbt_yes, lgbt_unknown + 4 source channels
    # Total dim: 1 (age) + 3 (gender) + 3 (lgbt) + 4 (sources) = 11 dims
    assert processor.demographics_dim == 11
    assert len(default_demo) == 11

    # User 1 is iPhone -> iPhone channel set to 1.0 (sources categories: [android, iphone, apple_watch, unknown])
    # index order: gender_male (index 1), gender_female (index 2), gender_unknown (index 3)
    # sources: android (index 7), iphone (index 8), apple_watch (index 9), unknown (index 10)
    assert demo_map[1][8] == 1.0  # iPhone source active
    assert demo_map[2][7] == 1.0  # Android source active


def test_prev_prediction_linker(dummy_splitter_data):
    train_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 1].copy()
    val_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 2].copy()
    test_df = dummy_splitter_data[dummy_splitter_data["app_user_id"] == 3].copy()

    t_df, v_df, test_res_df = PrevPredictionLinker.link(train_df, val_df, test_df)

    assert "prev_sample_idx" in t_df.columns
    assert "prev_sample_idx" in v_df.columns
    assert "prev_sample_idx" in test_res_df.columns

    # User 1 has 4 records, first should be -1, others point to previous index
    assert t_df.loc[0, "prev_sample_idx"] == -1
    assert t_df.loc[1, "prev_sample_idx"] == 0
    assert t_df.loc[2, "prev_sample_idx"] == 1
    assert t_df.loc[3, "prev_sample_idx"] == 2


def test_step_preprocessor():
    from src.data.components.preprocessing import StepPreprocessor
    
    # Create test data
    df = pd.DataFrame({
        "app_user_id": [22, 22, 26, 1],
        "start_timestamp": [
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00"
        ],
        "end_timestamp": [
            "2026-01-01 10:00:00", # 0 duration
            "2026-01-01 10:00:00", # 0 duration but steps = 0 (should keep, but android source drops it)
            "2026-01-01 10:05:00",
            "2026-01-01 10:05:00"
        ],
        "steps": [10, 0, 10, 10],
        "app_source": ["health.connect.android", "health.connect.android", "Health", "iPhone"]
    })
    
    preprocessor = StepPreprocessor()
    processed_df = preprocessor(df)
    
    # 1. 0 duration record with steps > 0 (user 22, row 0) should be dropped. 
    # Row 1 (steps=0) should be kept by duration check but dropped because app_user_id=22 contains android.
    # 2. iPhone source checks:
    # app_user_id 26: Health source mapped to Apple Watch.
    assert len(processed_df) == 2
    
    # Check remaining records
    processed_df = processed_df.reset_index(drop=True)
    assert processed_df.loc[0, "app_user_id"] == 26
    assert processed_df.loc[0, "app_source"] == "Apple Watch"
    assert processed_df.loc[1, "app_user_id"] == 1
    assert processed_df.loc[1, "app_source"] == "iPhone"


def test_calorie_preprocessor():
    from src.data.components.preprocessing import CaloriePreprocessor
    
    df = pd.DataFrame({
        "app_user_id": [1, 1, 1, 2, 2],
        "start_timestamp": [
            "2026-01-01 10:00:00", # study period (keep)
            "2026-01-01 10:00:00", # identical start/end
            "2025-08-01 10:00:00", # before study start (drop)
            "2026-01-01 10:00:00", # android conversion (>=3000 -> divide by 1000)
            "2026-01-01 10:00:00"  # duplicate of previous android record
        ],
        "end_timestamp": [
            "2026-01-01 10:05:00",
            "2026-01-01 10:00:00", # identical start/end (drop)
            "2025-08-01 10:05:00",
            "2026-01-01 10:05:00",
            "2026-01-01 10:05:00"
        ],
        "calories": [200.0, 100.0, 150.0, 3200.0, 3200.0],
        "app_source": ["iPhone", "iPhone", "iPhone", "health.connect.android", "health.connect.android"]
    })
    
    preprocessor = CaloriePreprocessor()
    processed_df = preprocessor(df)
    
    # Expected remaining rows:
    # 1. Row 0 (user 1, keep, calories = 200.0)
    # 2. Row 3 (user 2 android, keep and convert 3200.0 -> 3)
    # (Row 1 identical times dropped, Row 2 before study start dropped, Row 4 duplicate dropped)
    assert len(processed_df) == 2
    processed_df = processed_df.reset_index(drop=True)
    assert processed_df.loc[0, "app_user_id"] == 1
    assert processed_df.loc[0, "calories"] == 200.0
    assert processed_df.loc[1, "app_user_id"] == 2
    assert processed_df.loc[1, "calories"] == 3


@patch('src.data.components.cohort_builder.DatabaseService')
def test_cohort_builder_with_sleep_features(mock_db_class):
    """Tests CohortBuilder extracting sleep questions 54 and 55 as string and calculating features."""
    from src.data.components.cohort_builder import CohortBuilder
    from src.data.components.label_aggregators import MeanAggregator

    survey_df = pd.DataFrame({
        'id': [101, 102],
        'app_user_id': [10, 10],
        'timestamp': pd.to_datetime(['2025-10-01 08:00:00', '2025-10-01 09:00:00'])
    })

    answer_df = pd.DataFrame([
        # response 101 has asleep time
        {'survey_response_id': 101, 'question_id': 54, 'answer': "10:00 PM"},
        # response 102 has awake time
        {'survey_response_id': 102, 'question_id': 55, 'answer': "07:00 AM"},
        # and answer for question 2 (the aggregator target)
        {'survey_response_id': 101, 'question_id': 2, 'answer': "1"},
        {'survey_response_id': 102, 'question_id': 2, 'answer': "1"},
    ])

    dummy_data = {"step": pd.DataFrame(), "survey_response": survey_df, "answer": answer_df}

    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table].copy()

    agg = MeanAggregator(question_ids=[2])

    builder = CohortBuilder(
        modalities=[],
        modality_cols={},
        preprocessors=None,
        aggregator=agg,
        os_filter='both',
        use_sleep=True
    )

    _, master_df, _ = builder.build()

    # master_df should have 'sleep_hours' and 'sleep_category' columns merged
    assert 'sleep_hours' in master_df.columns
    assert 'sleep_category' in master_df.columns
    assert len(master_df) == 1
    assert master_df.loc[0, 'sleep_hours'] == 9.0
    assert master_df.loc[0, 'sleep_category'] == 1


@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_with_sleep_features(mock_db_class):
    """Tests HealthDataModule setup and HealthDataset context dimension with use_sleep=True."""
    from src.data.health_datamodule import HealthDataModule
    from src.data.components.label_aggregators import MeanAggregator
    from src.data.components.samplers import OffsetSampler

    # Define 4 users to support train/val/test splits without empty categories
    step_df = pd.DataFrame({
        'app_user_id': [10, 11, 12, 13],
        'start_timestamp': pd.to_datetime([
            '2025-10-01 02:00:00', '2025-10-01 02:00:00',
            '2025-10-01 02:00:00', '2025-10-01 02:00:00'
        ]),
        'steps': [100, 150, 200, 250]
    })

    survey_df = pd.DataFrame({
        'id': [101, 102, 103, 104],
        'app_user_id': [10, 11, 12, 13],
        'timestamp': pd.to_datetime([
            '2025-10-01 08:00:00', '2025-10-01 08:00:00',
            '2025-10-01 08:00:00', '2025-10-01 08:00:00'
        ])
    })

    answer_df = pd.DataFrame([
        # user 10: 9 hours sleep -> class 1
        {'survey_response_id': 101, 'question_id': 54, 'answer': "10:00 PM"},
        {'survey_response_id': 101, 'question_id': 55, 'answer': "07:00 AM"},
        {'survey_response_id': 101, 'question_id': 2, 'answer': "1"},

        # user 11: 4 hours sleep -> class 0
        {'survey_response_id': 102, 'question_id': 54, 'answer': "12:00 AM"},
        {'survey_response_id': 102, 'question_id': 55, 'answer': "04:00 AM"},
        {'survey_response_id': 102, 'question_id': 2, 'answer': "1"},

        # user 12: missing sleep data -> unknown sleep class
        {'survey_response_id': 103, 'question_id': 2, 'answer': "1"},

        # user 13: 11 hours sleep -> class 2
        {'survey_response_id': 104, 'question_id': 54, 'answer': "09:00 PM"},
        {'survey_response_id': 104, 'question_id': 55, 'answer': "08:00 AM"},
        {'survey_response_id': 104, 'question_id': 2, 'answer': "1"},
    ])

    demo_df = pd.DataFrame([
        {'app_user_id': 10, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 11, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 12, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 13, 'keyword': 'age', 'value': '40'},

        {'app_user_id': 10, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 11, 'keyword': 'gender identity', 'value': 'female'},
        {'app_user_id': 12, 'keyword': 'gender identity', 'value': 'male'},
        {'app_user_id': 13, 'keyword': 'gender identity', 'value': 'female'},

        {'app_user_id': 10, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 11, 'keyword': 'lgbt', 'value': 'yes'},
        {'app_user_id': 12, 'keyword': 'lgbt', 'value': 'no'},
        {'app_user_id': 13, 'keyword': 'lgbt', 'value': 'no'},
    ])

    dummy_data = {"step": step_df, "survey_response": survey_df, "answer": answer_df, "demographic": demo_df}

    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table]

    dm = HealthDataModule(

        exclude_user_ids=[],
        aggregator=MeanAggregator(question_ids=[2]),
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0),
        train_val_test_split=(0.5, 0.25, 0.25),
        use_demographics=True,
        use_sleep=True,
        modalities=["step"],
        # This fixture supplies no step records at all; the sleep features under
        # test come from the survey side, so sensor coverage is not required.
        require_sensor_data=False,
        # Isolate the sleep feature dimensions from per-response context features
        use_survey_context=False,
    )

    dm.setup()

    # We should have increased demographics_dim by 5 (base_demographics_dim=9 or 11 + 5 = 14 or 16)
    assert dm.demographics_dim in [14, 16]

    # Get the dataset item
    train_ds = dm.data_train
    assert len(train_ds) > 0
    sample = train_ds[0]

    # sample tuple elements:
    # 0: seq features
    # 1: target
    # 2: user_idx
    # 3: context vector (demographics + sleep features)
    assert len(sample) == 4
    context_vector = sample[3]

    # Context vector length should match dm.demographics_dim
    assert len(context_vector) == dm.demographics_dim


def test_distance_preprocessor():
    from src.data.components.preprocessing import DistancePreprocessor

    df = pd.DataFrame({
        "app_user_id": [1, 1, 1, 2, 2],
        "start_timestamp": [
            "2026-01-01 10:00:00",  # study period (keep)
            "2026-01-01 10:00:00",  # identical start/end (drop)
            "2025-08-01 10:00:00",  # before study start (drop)
            "2026-01-01 10:00:00",  # valid record (keep)
            "2026-01-01 10:00:00"   # duplicate of previous record (drop)
        ],
        "end_timestamp": [
            "2026-01-01 10:05:00",
            "2026-01-01 10:00:00",  # identical start/end
            "2025-08-01 10:05:00",
            "2026-01-01 10:05:00",
            "2026-01-01 10:05:00"
        ],
        "distance": [500.0, 100.0, 200.0, 1200.0, 1200.0],
        "app_source": ["iPhone", "iPhone", "iPhone", "health.connect.android", "health.connect.android"]
    })

    preprocessor = DistancePreprocessor()
    processed_df = preprocessor(df)

    # Expected remaining rows:
    # 1. Row 0 (user 1, keep, distance = 500.0)
    # 2. Row 3 (user 2 android, keep, distance = 1200.0)
    assert len(processed_df) == 2
    processed_df = processed_df.reset_index(drop=True)
    assert processed_df.loc[0, "app_user_id"] == 1
    assert processed_df.loc[0, "distance"] == 500.0
    assert processed_df.loc[1, "app_user_id"] == 2
    assert processed_df.loc[1, "distance"] == 1200.0


@patch('src.data.components.cohort_builder.DatabaseService')
def test_datamodule_with_distance_modality(mock_db_class):
    """Tests HealthDataModule setup with distance modality and DistancePreprocessor."""
    from src.data.health_datamodule import HealthDataModule
    from src.data.components.label_aggregators import MeanAggregator
    from src.data.components.samplers import OffsetSampler
    from src.data.components.preprocessing import DistancePreprocessor

    step_df = pd.DataFrame({
        'app_user_id': [10, 11, 12, 13],
        'start_timestamp': pd.to_datetime([
            '2025-10-01 02:00:00', '2025-10-01 02:00:00',
            '2025-10-01 02:00:00', '2025-10-01 02:00:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2025-10-01 03:00:00', '2025-10-01 03:00:00',
            '2025-10-01 03:00:00', '2025-10-01 03:00:00'
        ]),
        'steps': [100, 150, 200, 250],
        'app_source': ['iPhone'] * 4
    })

    distance_df = pd.DataFrame({
        'app_user_id': [10, 11, 12, 13],
        'start_timestamp': pd.to_datetime([
            '2025-10-01 02:00:00', '2025-10-01 02:00:00',
            '2025-10-01 02:00:00', '2025-10-01 02:00:00'
        ]),
        'end_timestamp': pd.to_datetime([
            '2025-10-01 03:00:00', '2025-10-01 03:00:00',
            '2025-10-01 03:00:00', '2025-10-01 03:00:00'
        ]),
        'distance': [500.0, 750.0, 1000.0, 1250.0],
        'app_source': ['iPhone'] * 4
    })

    survey_df = pd.DataFrame({
        'id': [101, 102, 103, 104],
        'app_user_id': [10, 11, 12, 13],
        'timestamp': pd.to_datetime([
            '2025-10-01 08:00:00', '2025-10-01 08:00:00',
            '2025-10-01 08:00:00', '2025-10-01 08:00:00'
        ])
    })

    answer_df = pd.DataFrame([
        {'survey_response_id': 101, 'question_id': 2, 'answer': "1"},
        {'survey_response_id': 102, 'question_id': 2, 'answer': "1"},
        {'survey_response_id': 103, 'question_id': 2, 'answer': "1"},
        {'survey_response_id': 104, 'question_id': 2, 'answer': "1"},
    ])

    demo_df = pd.DataFrame([
        {'app_user_id': 10, 'keyword': 'age', 'value': '25'},
        {'app_user_id': 11, 'keyword': 'age', 'value': '30'},
        {'app_user_id': 12, 'keyword': 'age', 'value': '35'},
        {'app_user_id': 13, 'keyword': 'age', 'value': '40'},
    ])

    dummy_data = {
        "step": step_df,
        "distance": distance_df,
        "survey_response": survey_df,
        "answer": answer_df,
        "demographic": demo_df
    }

    mock_db = mock_db_class.return_value
    mock_db.connect.return_value = True
    mock_db.extract_from_database.side_effect = lambda table: dummy_data[table].copy()

    dm = HealthDataModule(
        aggregator=MeanAggregator(question_ids=[2]),
        sampler=OffsetSampler(start_offset_hours=-24, end_offset_hours=0, include_time_features=False),
        train_val_test_split=(0.5, 0.25, 0.25),
        modalities=["step", "distance"],
        preprocessors={"distance": DistancePreprocessor()},
        require_sensor_data=False
    )

    dm.setup()

    assert dm.data_train is not None
    assert len(dm.data_train) > 0
    sample = dm.data_train[0]
    # Check sequence shape: [24 hours, 2 modalities]
    assert sample[0].shape == (24, 2)




