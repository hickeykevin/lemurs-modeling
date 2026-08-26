from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from src.data.components.cohort_splitter import (
    CohortSplitter,
    WalkForwardFold,
    lookback_hours_from_sampler,
)
from src.data.components.health_dataset import HealthDataset
from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
from src.data.health_datamodule import HealthDataModule


class WalkForwardHealthDataModule(HealthDataModule):
    """Per-user, purged, expanding-window walk-forward cross-validation.

    Answers "given a user's own history, can the model forecast their
    near-future risk?" — the question this study is centered on. This is a
    different question from ``CVHealthDataModule``'s user-grouped k-fold
    (which measures generalization to a *new* user with no prior history —
    a zero-shot deployment-day question) and a statistically stronger version
    of ``split_mode="longitudinal"``'s single train/val/test cut (which
    samples the forecasting question exactly once per user; this module
    samples it multiple times per user, walking forward through their
    timeline, and pools predictions across users and folds).

    Each fold's train set is everything before its test window — all of a
    user's own past, not a fixed-size trailing slice, since the model is
    meant to accumulate history the way a deployed app would. Every fold
    boundary is purged exactly as ``CohortSplitter``'s other modes purge
    their single boundary; see ``CohortSplitter.split_walk_forward`` for the
    full mechanics and the fold-alignment caveat (fold index is a per-user
    position, not a shared calendar window across users).

    A user contributes as many folds as their response count supports after
    ``burn_in_responses``; short-history users contribute fewer folds, or
    none, and this is expected rather than an error (see
    ``CohortSplitter.split_walk_forward``). Set ``burn_in_responses`` from
    the cohort's actual response-count distribution, not a fixed guess — see
    the ``configs/data/walk_forward.yaml`` comment for the numbers this
    cohort's distribution supports.

    ``current_fold`` selects which of the precomputed folds ``setup()``
    builds datasets for, in the same style as ``CVHealthDataModule``. Unlike
    that module, this one does not repeat with different seeds
    (``num_repeats``/``current_repeat`` do not apply here): a walk-forward
    fold sequence is not an independently-reshuffled draw the way a grouped
    k-fold's fold assignment is, so re-running it with a different seed
    would not answer a new question the way ``CVHealthDataModule``'s repeats
    do.

    ``data_val`` and ``data_test`` are built with ``return_index=True`` so a
    ``PredictionCollectorCallback`` can join predictions back to
    ``data_links`` (``app_user_id``, ``record_timestamp``) — needed to pool
    predictions across folds and users into one evaluation set. This is the
    one respect in which this module cannot simply override ``_split_data``
    the way ``CVHealthDataModule`` does and leave the rest of ``setup()``
    untouched: the base ``setup()`` never passes ``return_index`` to
    ``HealthDataset``. Rather than duplicating all of ``setup()``, this
    module calls ``super().setup(stage)`` first — building ``data_train``
    with the fold's data exactly as the base class would — then rebuilds
    only ``data_val``/``data_test`` with ``return_index=True``, reusing the
    scaler, ``demographics_map``, and ``user_to_idx`` the base class already
    fit, so nothing about the fitted state is redone or diverges from what
    ``_split_data`` selected.
    """

    def __init__(
        self,
        aggregator: LabelAggregator,
        sampler: TimeSampler,
        burn_in_responses: Optional[int] = None,
        step_responses: Optional[int] = None,
        val_responses: int = 0,
        fold_sizing: Literal["count", "pct", "cyclic"] = "count",
        burn_in_pct: Optional[float] = None,
        step_pct: Optional[float] = None,
        val_pct: float = 0.0,
        train_width_pct: Optional[float] = None,
        current_fold: int = 0,
        scaler: Optional[Any] = None,
        preprocessors: Optional[Dict[str, Any]] = None,
        modalities: List[str] = ["step"],
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_state: int = 42,
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "none",
        use_demographics: bool = True,
        use_sleep: bool = False,
        enrollment_lead_days: float = 7.0,
        enrollment_trail_days: float = 1.0,
        require_sensor_data: bool = True,
        use_survey_context: bool = True,
        include_time_features: Optional[bool] = None,
        exclude_user_ids: Optional[List[int]] = None,
        purge_hours: Optional[float] = None,
        prebuilt_cohort: Optional[Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]] = None,
    ) -> None:
        """Initializes the WalkForwardHealthDataModule.

        Args:
            fold_sizing: Which ``CohortSplitter`` fold-construction method to
                use. ``"count"`` (default): fixed *response count* per fold
                (``CohortSplitter.split_walk_forward``, driven by
                ``burn_in_responses``/``step_responses``/``val_responses``) --
                every fold's test window is the same absolute size, so users
                with less remaining history stop contributing to later folds
                one by one, and the last fold or two can end up with only a
                handful of users. ``"pct"``: fixed *fraction of each user's
                own total responses* per fold, still an EXPANDING train
                window from the user's earliest response
                (``CohortSplitter.split_walk_forward_pct``, driven by
                ``burn_in_pct``/``step_pct``/``val_pct``) -- every eligible
                user contributes to every fold this method produces, since
                test size scales with their own total rather than a shared
                absolute count. ``"cyclic"``: fixed *train width* that slides
                (not expands) through a user's full response range, wrapping
                past the end back to the start, so every response is tested
                on exactly once per cycle and a user's most-recent responses
                can train a fold whose test window sits at the very start of
                their timeline (``CohortSplitter.split_walk_forward_cyclic``,
                driven by ``train_width_pct``/``step_pct``; no val-window
                concept). See that method's docstring for when this differs
                from ``"pct"`` in a way that matters for what you're
                measuring -- it trades deployment realism (never training on
                a user's chronological future) for folds of uniform size and
                difficulty.
            burn_in_responses: ``fold_sizing="count"`` only. Minimum number of
                a user's earliest responses reserved for the first fold's
                training set before any forecasting is evaluated for that
                user. See ``CohortSplitter.split_walk_forward``.
            step_responses: ``fold_sizing="count"`` only. Number of responses
                each successive fold's test window covers, and by which the
                train window expands.
            val_responses: ``fold_sizing="count"`` only. Number of responses
                each fold's validation window covers. Defaults to 0.
            burn_in_pct: ``fold_sizing="pct"`` only. Fraction (0 < x < 1) of a
                user's earliest responses reserved for the first fold's
                training set. See ``CohortSplitter.split_walk_forward_pct``.
            step_pct: ``fold_sizing="pct"`` only. Fraction of a user's total
                responses each successive fold's test window covers.
            val_pct: ``fold_sizing="pct"`` only. Fraction of a user's total
                responses each fold's validation window covers. Defaults to
                0.0.
            train_width_pct: ``fold_sizing="cyclic"`` only. Fraction (0 < x < 1)
                of a user's total responses in every fold's fixed-width train
                window. See ``CohortSplitter.split_walk_forward_cyclic``.
            (``val_responses``/``val_pct``, whichever applies): Defaults to
                no validation window at all -- every fold is just
                train | purge | test, no rows are set aside for early
                stopping / checkpoint selection, ``data_val`` is an empty
                dataset every fold, and (with no ``val/*`` metric ever
                logged) ``EarlyStopping``/``ModelCheckpoint`` never have
                anything to monitor -- a caller should expect every fold to
                train for its full configured epoch budget and evaluate
                whatever weights exist at the end of training.
            current_fold: Which walk-forward fold (0-indexed, chronological)
                to build ``data_train``/``data_val``/``data_test`` for. An
                orchestration script loops this the way ``cv_train.py`` loops
                ``current_fold`` for ``CVHealthDataModule`` — see
                ``get_num_folds()`` for how many folds exist for this cohort
                and configuration.
            purge_hours: Width of the gap purged around every fold's
                train/test boundary (or train/val and val/test, when a val
                window is configured). Defaults to the sampler's own
                lookback window when None, same as the base class's
                "longitudinal" mode — see ``lookback_hours_from_sampler``.

        See ``HealthDataModule`` for all other arguments.
        """
        if fold_sizing == "count":
            if burn_in_responses is None or step_responses is None:
                raise ValueError(
                    "fold_sizing='count' requires burn_in_responses and "
                    "step_responses to be set."
                )
        elif fold_sizing == "pct":
            if burn_in_pct is None or step_pct is None:
                raise ValueError(
                    "fold_sizing='pct' requires burn_in_pct and step_pct to be set."
                )
        elif fold_sizing == "cyclic":
            if train_width_pct is None or step_pct is None:
                raise ValueError(
                    "fold_sizing='cyclic' requires train_width_pct and step_pct to be set."
                )
        else:
            raise ValueError(
                f"fold_sizing must be 'count', 'pct', or 'cyclic'; got {fold_sizing!r}"
            )

        super().__init__(
            aggregator=aggregator,
            sampler=sampler,
            scaler=scaler,
            preprocessors=preprocessors,
            modalities=modalities,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_val_test_split=(0.0, 0.0, 0.0),  # unused: walk-forward derives its own cuts
            random_state=random_state,
            split_mode="longitudinal",  # unused by this module's _split_data override, but
            # must be a value HealthDataModule.__init__ accepts.
            os_filter=os_filter,
            collapse_strategy=collapse_strategy,
            use_demographics=use_demographics,
            use_sleep=use_sleep,
            enrollment_lead_days=enrollment_lead_days,
            enrollment_trail_days=enrollment_trail_days,
            require_sensor_data=require_sensor_data,
            use_survey_context=use_survey_context,
            include_time_features=include_time_features,
            exclude_user_ids=exclude_user_ids,
            prebuilt_cohort=prebuilt_cohort,
            purge_hours=purge_hours,
        )
        self.hparams.burn_in_responses = burn_in_responses
        self.hparams.step_responses = step_responses
        self.hparams.val_responses = val_responses
        self.hparams.fold_sizing = fold_sizing
        self.hparams.burn_in_pct = burn_in_pct
        self.hparams.step_pct = step_pct
        self.hparams.val_pct = val_pct
        self.hparams.train_width_pct = train_width_pct
        self.hparams.current_fold = current_fold
        # Re-save hyperparameters to capture the walk-forward-specific fields
        self.save_hyperparameters(logger=False)

        self._folds: Optional[List[WalkForwardFold]] = None

    def get_num_folds(self) -> int:
        """Returns how many walk-forward folds this cohort/config produces.

        Requires building the cohort and computing every fold up front (the
        fold count depends on how many of each user's responses clear
        burn-in), so this triggers ``setup()`` if it has not run yet — same
        cost ``CVHealthDataModule.get_num_folds()`` pays for the same reason.
        """
        if not hasattr(self, "master_df"):
            self.setup()
        return len(self._get_folds(self.master_df))

    def _get_folds(self, df: pd.DataFrame) -> List[WalkForwardFold]:
        """Computes and caches every walk-forward fold for ``df``.

        Cached on the instance because ``_split_data`` (called once per
        ``setup()``) and ``get_num_folds`` both need the full fold list, and
        recomputing it is pure redundant work — the fold list does not
        depend on ``current_fold``.
        """
        if self._folds is None:
            purge_hours = self.hparams.purge_hours
            if purge_hours is None:
                purge_hours = lookback_hours_from_sampler(self.hparams.sampler)

            splitter = CohortSplitter(purge_hours=purge_hours)
            if self.hparams.fold_sizing == "pct":
                self._folds = splitter.split_walk_forward_pct(
                    df,
                    burn_in_pct=self.hparams.burn_in_pct,
                    step_pct=self.hparams.step_pct,
                    val_pct=self.hparams.val_pct,
                )
            elif self.hparams.fold_sizing == "cyclic":
                self._folds = splitter.split_walk_forward_cyclic(
                    df,
                    train_width_pct=self.hparams.train_width_pct,
                    step_pct=self.hparams.step_pct,
                )
            else:
                self._folds = splitter.split_walk_forward(
                    df,
                    burn_in_responses=self.hparams.burn_in_responses,
                    step_responses=self.hparams.step_responses,
                    val_responses=self.hparams.val_responses,
                )
        return self._folds

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns the (train, val, test) triple for ``self.hparams.current_fold``."""
        folds = self._get_folds(df)
        if not folds:
            if self.hparams.fold_sizing == "pct":
                params_msg = (
                    f"burn_in_pct={self.hparams.burn_in_pct} + "
                    f"val_pct={self.hparams.val_pct} + step_pct={self.hparams.step_pct}"
                )
            elif self.hparams.fold_sizing == "cyclic":
                params_msg = (
                    f"train_width_pct={self.hparams.train_width_pct} + "
                    f"step_pct={self.hparams.step_pct}"
                )
            else:
                params_msg = (
                    f"burn_in_responses={self.hparams.burn_in_responses} + "
                    f"val_responses={self.hparams.val_responses} + "
                    f"step_responses={self.hparams.step_responses}"
                )
            raise MisconfigurationException(
                f"No user in this cohort has enough responses to clear {params_msg}. "
                "Lower these, or check the cohort's response-count distribution."
            )
        fold_idx = self.hparams.current_fold
        if not (0 <= fold_idx < len(folds)):
            raise MisconfigurationException(
                f"current_fold={fold_idx} out of range: this cohort/config "
                f"produces {len(folds)} walk-forward fold(s) (0..{len(folds) - 1})."
            )
        fold = folds[fold_idx]
        return fold.train_df, fold.val_df, fold.test_df

    def setup(self, stage: Optional[str] = None) -> None:
        """Builds datasets for ``current_fold``, with index-returning val/test sets.

        Delegates to the base class for the entire cohort-extraction, split,
        scaler-fitting and dataset-construction pipeline (identical to
        ``split_mode="longitudinal"`` in every respect except which rows
        ``_split_data`` selects), then rebuilds ``data_val``/``data_test``
        only, with ``return_index=True``, reusing the scaler/demographics
        state the base class already fit. See the class docstring for why
        this is necessary rather than overriding ``_split_data`` alone.
        """
        already_built = self.data_train is not None or self.data_val is not None or self.data_test is not None
        super().setup(stage)
        if already_built:
            return  # base class no-ops on repeat calls; so do we

        is_regression = getattr(self.hparams.aggregator, "is_regression", False)
        modality_dfs = self.modality_dfs

        for attr, df in (("data_val", self.data_val.data_links), ("data_test", self.data_test.data_links)):
            setattr(
                self,
                attr,
                HealthDataset(
                    df, modality_dfs, self.hparams.modality_cols,
                    self.hparams.sampler, self.hparams.scaler, user_to_idx=self.user_to_idx,
                    is_regression=is_regression,
                    demographics_map=self.demographics_map,
                    default_demographics=self.default_demographics,
                    use_sleep=self.hparams.use_sleep,
                    use_survey_context=self.hparams.use_survey_context,
                    return_index=True,
                ),
            )
