from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from src.data.components.cohort_splitter import (
    CohortSplitter,
    WalkForwardFold,
    lookback_hours_from_sampler,
)
from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
from src.data.indexed_health_datamodule import IndexedHealthDataModule


class WalkForwardHealthDataModule(IndexedHealthDataModule):
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

    Under ``fold_sizing="pct"`` each fold's train set is everything before
    its test window — all of a user's own past, not a fixed-size trailing
    slice, since the model is meant to accumulate history the way a deployed
    app would. Under ``"cyclic"`` the train window is instead a fixed width
    that slides and wraps. Every fold boundary is purged exactly as
    ``CohortSplitter``'s other modes purge their single boundary; see the
    two ``split_walk_forward_*`` methods for the full mechanics and the
    fold-alignment caveat (fold index is a per-user position, not a shared
    calendar window across users).

    Both modes size folds as a fraction of each user's own total responses,
    so every eligible user contributes to every fold. An earlier
    ``fold_sizing="count"`` mode sized them by absolute response count
    instead, which made short-history users drop out of later folds one by
    one; it was removed once no config used it (see git history).

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
    predictions across folds and users into one evaluation set. This is why
    this class subclasses ``IndexedHealthDataModule`` rather than
    ``HealthDataModule`` directly: ``IndexedHealthDataModule.setup()``
    already does exactly the "run the normal setup, then rebuild
    data_val/data_test with return_index=True, reusing the fitted
    scaler/demographics state" work this module also needs, so this class
    inherits that ``setup()`` unchanged and only overrides ``_split_data``
    to select ``current_fold``'s rows before the inherited ``setup()`` runs.
    """

    def __init__(
        self,
        aggregator: LabelAggregator,
        sampler: TimeSampler,
        fold_sizing: Literal["pct", "cyclic"] = "cyclic",
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
                use. ``"pct"``: fixed *fraction of each user's
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
            val_pct (when it applies): Defaults to
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
        if fold_sizing == "pct":
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
                f"fold_sizing must be 'pct' or 'cyclic'; got {fold_sizing!r}"
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
            random_state=random_state,
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
            else:
                self._folds = splitter.split_walk_forward_cyclic(
                    df,
                    train_width_pct=self.hparams.train_width_pct,
                    step_pct=self.hparams.step_pct,
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
            else:
                params_msg = (
                    f"train_width_pct={self.hparams.train_width_pct} + "
                    f"step_pct={self.hparams.step_pct}"
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

    # setup() is inherited unchanged from IndexedHealthDataModule: it calls
    # super().setup(stage), which (via HealthDataModule.setup()) dispatches
    # to THIS class's _split_data override above to select current_fold's
    # rows, then rebuilds data_val/data_test with return_index=True. No
    # override needed here -- see the class docstring.
