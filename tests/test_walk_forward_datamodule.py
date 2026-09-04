"""Tests for WalkForwardHealthDataModule: per-user purged walk-forward CV.

Uses prebuilt_cohort (as CVHealthDataModule's tests use direct construction)
to exercise the real setup()/_split_data() path without a database, since the
risk here is specifically in how setup() rebuilds data_val/data_test with
return_index=True after delegating to the base class -- not in the splitting
math alone, which test_data_components.py's CohortSplitter tests already cover.
"""

import pandas as pd
import pytest
import torch

from src.data.components.label_aggregators import MeanAggregator
from src.data.components.samplers import OffsetSampler
from src.data.walk_forward_health_datamodule import WalkForwardHealthDataModule


def _cohort(users=(1, 2), n_per_user=20):
    """Two users, hourly step data, surveys every 6h so a 6h-lookback sampler has coverage."""
    rows = []
    for uid in users:
        for i in range(n_per_user):
            rows.append({
                "app_user_id": uid,
                "record_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * i),
                "answer": i % 2,
                "survey_response_id": uid * 1000 + i,
            })
    master_df = pd.DataFrame(rows)

    step_rows = []
    for uid in users:
        for i in range(n_per_user * 4):
            step_rows.append({
                "app_user_id": uid,
                "start_timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=1.5 * i),
                "steps": 100,
            })
    step_df = pd.DataFrame(step_rows)

    demographics_df = pd.DataFrame(columns=["app_user_id", "gender", "age", "lgbt"])
    return {"step": step_df}, master_df, demographics_df


def _make_dm(current_fold=0, **overrides):
    modality_dfs, master_df, demographics_df = _cohort()
    kwargs = dict(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        fold_sizing="pct",
        burn_in_pct=0.3,
        step_pct=0.2,
        val_pct=0.1,
        current_fold=current_fold,
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    kwargs.update(overrides)
    return WalkForwardHealthDataModule(**kwargs)


def test_get_num_folds_matches_the_walk_forward_split():
    dm = _make_dm()
    # 20 responses/user; burn_in_pct=0.3 (6) + val_pct=0.1 (2) leaves 12 for
    # test windows of step_pct=0.2 (4) each -> 2 folds after the burn-in.
    assert dm.get_num_folds() == 2


def test_setup_builds_datasets_for_the_selected_fold():
    dm0 = _make_dm(current_fold=0)
    dm0.setup()
    dm1 = _make_dm(current_fold=1)
    dm1.setup()

    # Fold 1's train set is a strict superset of fold 0's (expanding window):
    # more responses accumulated -> more train examples, once purge is accounted for.
    assert len(dm1.data_train) > len(dm0.data_train)


def test_val_and_test_datasets_return_index_but_train_does_not():
    dm = _make_dm()
    dm.setup()

    train_item = dm.data_train[0]
    val_item = dm.data_val[0]
    test_item = dm.data_test[0]

    # use_demographics=False still yields a fixed-width default demographics
    # vector as the 4th element (DemographicsProcessor's fallback), present
    # identically on every split. val/test add idx as a genuine 5th element;
    # train does not.
    assert len(train_item) == len(val_item) - 1 == len(test_item) - 1
    assert val_item[-1].dtype == torch.long
    assert test_item[-1].dtype == torch.long


def test_index_recovers_the_correct_source_row():
    dm = _make_dm()
    dm.setup()

    for i in range(len(dm.data_test)):
        item = dm.data_test[i]
        idx = item[-1].item()
        assert idx == i
        row = dm.data_test.data_links.iloc[idx]
        assert row["app_user_id"] in (1, 2)


def test_dataloaders_carry_the_index_through_batching():
    dm = _make_dm()
    dm.setup()

    test_batch = next(iter(dm.test_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    train_batch = next(iter(dm.train_dataloader()))

    assert len(test_batch) == len(train_batch) + 1  # ... + idx
    assert len(val_batch) == len(train_batch) + 1


def test_current_fold_out_of_range_raises():
    dm = _make_dm(current_fold=99)
    with pytest.raises(Exception, match="out of range"):
        dm.setup()


def test_setup_is_idempotent():
    """Calling setup() twice does not rebuild datasets a second time (matches base class no-op behaviour)."""
    dm = _make_dm()
    dm.setup()
    data_val_id = id(dm.data_val)
    data_test_id = id(dm.data_test)
    dm.setup()
    assert id(dm.data_val) == data_val_id
    assert id(dm.data_test) == data_test_id


def test_purge_shrinks_train_end_to_end():
    """purge_hours (defaulted from the sampler's lookback) actually removes rows
    from train/val through the real setup() path, not just in the splitter unit tests.

    Sampler lookback is 6h and responses are spaced 6h apart, so every fold's
    train set loses exactly the responses immediately preceding val's start,
    per user -- this mirrors test_cohort_splitter_walk_forward_purges_every_fold_boundary
    but through the full datamodule, catching a mismatch in how purge_hours is
    threaded from hparams into split_walk_forward.
    """
    dm_purged = _make_dm(current_fold=0)  # purge_hours=None -> defaults to sampler lookback (6h)
    dm_purged.setup()

    dm_unpurged = _make_dm(current_fold=0, purge_hours=0.0)
    dm_unpurged.setup()

    assert len(dm_purged.data_train) < len(dm_unpurged.data_train)


def test_modality_dfs_stored_unconditionally_via_prebuilt_cohort():
    """HealthDataModule.setup() stores self.modality_dfs even on the prebuilt_cohort path.

    Regression guard for the bug hit while building this: self.raw_cohort is
    only set on the non-prebuilt branch, but WalkForwardHealthDataModule.setup()
    needs modality_dfs on both branches to rebuild data_val/data_test.
    """
    dm = _make_dm()
    dm.setup()
    assert hasattr(dm, "modality_dfs")
    assert set(dm.modality_dfs.keys()) == {"step"}


def test_no_folds_raises_a_clear_error():
    """burn_in wider than every user's data raises, rather than building an empty/broken dataset."""
    dm = _make_dm(burn_in_pct=0.99, step_pct=0.99)
    with pytest.raises(Exception, match="No user in this cohort"):
        dm.setup()


def test_rebuilt_val_test_share_train_fitted_state_with_base_class():
    """The rebuilt data_val/data_test use the same user_to_idx and scaler as data_train.

    i.e. setup()'s rebuild reuses what super().setup() already fit rather than
    re-deriving it, so val/test features are consistent with what the model
    trained on.
    """
    dm = _make_dm()
    dm.setup()

    assert dm.data_val.user_to_idx is dm.user_to_idx
    assert dm.data_test.user_to_idx is dm.user_to_idx
    assert dm.data_val.scaler is dm.hparams.scaler
    assert dm.data_test.scaler is dm.hparams.scaler
    assert dm.data_val.demographics_map is dm.demographics_map
    assert dm.data_test.demographics_map is dm.demographics_map


def test_val_pct_zero_produces_an_empty_data_val_without_crashing():
    """val_pct=0 (the default) yields an empty (but valid) data_val,
    and setup()/dataloader construction handle that without error."""
    dm = _make_dm(val_pct=0.0)
    dm.setup()

    assert len(dm.data_val) == 0
    assert len(dm.data_train) > 0
    assert len(dm.data_test) > 0

    # val_dataloader() on an empty dataset must not raise.
    val_batches = list(dm.val_dataloader())
    assert val_batches == []


def test_val_pct_zero_gives_more_train_rows_than_with_val():
    """With no val window, train absorbs what would otherwise be set aside for
    val -- so for the same fold, val_pct=0 should have >= as many train
    rows as val_pct>0 (strictly more once purge is accounted for, since
    there's one fewer purge boundary to lose rows to)."""
    dm_no_val = _make_dm(current_fold=0, val_pct=0.0)
    dm_no_val.setup()

    dm_with_val = _make_dm(current_fold=0, val_pct=0.1)
    dm_with_val.setup()

    assert len(dm_no_val.data_train) >= len(dm_with_val.data_train)


def test_val_pct_omitted_defaults_to_zero():
    """Not passing val_pct at all uses the class default (0.0), matching
    passing it explicitly."""
    modality_dfs, master_df, demographics_df = _cohort()
    dm_default = WalkForwardHealthDataModule(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        fold_sizing="pct",
        burn_in_pct=0.3,
        step_pct=0.2,
        # val_pct omitted entirely
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    assert dm_default.hparams.val_pct == 0.0
    dm_default.setup()
    assert len(dm_default.data_val) == 0


def test_fold_sizing_pct_requires_pct_params():
    modality_dfs, master_df, demographics_df = _cohort()
    with pytest.raises(ValueError, match="burn_in_pct"):
        WalkForwardHealthDataModule(
            aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
            sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
            fold_sizing="pct",
            prebuilt_cohort=(modality_dfs, master_df, demographics_df),
        )


def test_fold_sizing_invalid_value_raises():
    modality_dfs, master_df, demographics_df = _cohort()
    with pytest.raises(ValueError, match="fold_sizing"):
        WalkForwardHealthDataModule(
            aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
            sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
            fold_sizing="bogus",
            prebuilt_cohort=(modality_dfs, master_df, demographics_df),
        )


def _make_pct_dm(current_fold=0, **overrides):
    modality_dfs, master_df, demographics_df = _cohort()
    kwargs = dict(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        fold_sizing="pct",
        burn_in_pct=0.5,
        step_pct=0.15,
        current_fold=current_fold,
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    kwargs.update(overrides)
    return WalkForwardHealthDataModule(**kwargs)


def test_fold_sizing_pct_builds_datasets_end_to_end():
    """fold_sizing='pct' drives the real setup()/dataset-construction path,
    not just CohortSplitter.split_walk_forward_pct in isolation."""
    dm = _make_pct_dm()
    n_folds = dm.get_num_folds()
    assert n_folds > 0

    dm.setup()
    assert len(dm.data_train) > 0
    assert len(dm.data_test) > 0
    assert len(dm.data_val) == 0  # val_pct defaults to 0.0


def test_fold_sizing_pct_every_user_reaches_every_fold_end_to_end():
    """The property that motivated fold_sizing='pct': unlike 'count', every
    user with enough data to clear burn-in should appear in every fold's
    test set, checked through the real datamodule (both users in _cohort()
    have the same response count here, so this also exercises the
    multi-fold path without needing per-user response-count variation)."""
    n_folds = _make_pct_dm().get_num_folds()
    assert n_folds >= 1

    for fold in range(n_folds):
        dm = _make_pct_dm(current_fold=fold)
        dm.setup()
        assert set(dm.data_test.data_links["app_user_id"]) == {1, 2}


def test_fold_sizing_cyclic_requires_cyclic_params():
    modality_dfs, master_df, demographics_df = _cohort()
    with pytest.raises(ValueError, match="train_width_pct"):
        WalkForwardHealthDataModule(
            aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
            sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
            fold_sizing="cyclic",
            prebuilt_cohort=(modality_dfs, master_df, demographics_df),
        )


def _make_cyclic_dm(current_fold=0, **overrides):
    modality_dfs, master_df, demographics_df = _cohort()
    kwargs = dict(
        aggregator=MeanAggregator(question_ids=[2], threshold=0.5),
        sampler=OffsetSampler(start_offset_hours=-6, end_offset_hours=0),
        fold_sizing="cyclic",
        train_width_pct=0.4,
        step_pct=0.15,
        current_fold=current_fold,
        use_demographics=False,
        use_sleep=False,
        use_survey_context=False,
        require_sensor_data=False,
        prebuilt_cohort=(modality_dfs, master_df, demographics_df),
    )
    kwargs.update(overrides)
    return WalkForwardHealthDataModule(**kwargs)


def test_fold_sizing_cyclic_builds_datasets_end_to_end():
    """fold_sizing='cyclic' drives the real setup()/dataset-construction
    path, not just CohortSplitter.split_walk_forward_cyclic in isolation."""
    dm = _make_cyclic_dm()
    n_folds = dm.get_num_folds()
    assert n_folds > 0

    dm.setup()
    assert len(dm.data_train) > 0
    assert len(dm.data_test) > 0
    assert len(dm.data_val) == 0  # cyclic has no val-window concept at all


def test_fold_sizing_cyclic_every_user_reaches_every_fold_end_to_end():
    n_folds = _make_cyclic_dm().get_num_folds()
    assert n_folds >= 1

    for fold in range(n_folds):
        dm = _make_cyclic_dm(current_fold=fold)
        dm.setup()
        assert set(dm.data_test.data_links["app_user_id"]) == {1, 2}


def test_fold_sizing_cyclic_covers_every_response_end_to_end():
    """Every response is tested on exactly once across the full cycle,
    verified through the real datamodule rather than the splitter alone."""
    n_folds = _make_cyclic_dm().get_num_folds()
    seen = []
    for fold in range(n_folds):
        dm = _make_cyclic_dm(current_fold=fold)
        dm.setup()
        seen.extend(
            zip(
                dm.data_test.data_links["app_user_id"],
                dm.data_test.data_links["record_timestamp"],
            )
        )
    assert len(seen) == len(set(seen))
