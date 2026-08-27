import math
from typing import Tuple, Literal, Any, List, NamedTuple
import pandas as pd
from sklearn.model_selection import train_test_split


class WalkForwardFold(NamedTuple):
    """One fold of a per-user walk-forward split.

    ``fold_index`` is the fold's position in the walk (0-based, chronological)
    so callers can report per-fold diagnostics (e.g. "does accuracy improve
    with more accumulated history") without re-deriving order from timestamps.
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    fold_index: int


def lookback_hours_from_sampler(sampler: Any) -> float:
    """Derives a purge width (in hours) from a ``TimeSampler``'s own window.

    Reads the sampler's ``window_bounds`` at an arbitrary anchor timestamp
    rather than a type-specific attribute (``lookback_hours``,
    ``lookback_days``, ``bin_edges_hours``, ...), so it works uniformly
    across every ``TimeSampler`` subclass, including future ones. Samplers
    that read no sensor data (e.g. ``LagSampler``) report no window and
    contribute 0 — there is nothing to purge for them.
    """
    if not hasattr(sampler, "window_bounds"):
        return 0.0
    bounds = sampler.window_bounds(pd.Timestamp("2000-01-01"))
    if bounds is None:
        return 0.0
    start, end = bounds
    return max(0.0, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600.0)


class CohortSplitter:
    """Handles splitting the cohort dataset into train, validation, and test sets.

    Supports two strategies:
        - "user": Split by user ID (disjoint populations).
        - "longitudinal": Temporal split per user (predict future from past).

    Row-level random splitting is deliberately not offered. The label in this
    study is largely a person-level trait: 27 of 44 users are never positive,
    four are positive on more than 70% of their responses, and five users hold
    63% of all positives. Splitting rows at random therefore puts the same
    person on both sides of the split, and a model only has to recognise *who*
    a sequence belongs to — which activity streams make easy — to score well.
    The resulting metric is an identity-recognition score wearing a risk
    prediction label. It is also the wrong question: new users of the app never
    take surveys, so every deployment prediction is for a person the model has
    no labels for, which is exactly what a user-level split measures.

    Longitudinal mode has its own, separate leakage channel: it partitions
    survey *responses* by chronological position, but the raw sensor tables
    (``modality_dfs``) are shared unmodified across train/val/test. A sampler
    with a lookback window reads backward from each response's timestamp, so
    a val or test response sitting close in time to the preceding split's
    last response has a window that overlaps — and can contain sensor records
    identical to — the window used to build a sample on the other side of the
    boundary. Left unpurged, this lets a model see raw inputs from a "held
    out" sample during training, independent of any genuine same-user
    future-prediction skill. ``purge_hours`` closes this by dropping the
    responses closest to each boundary rather than assigning every response
    to a side.
    """

    #: Split modes that were removed, mapped to why, so configs fail loudly
    #: rather than silently falling back to something else.
    REMOVED_SPLIT_MODES = {
        "random": (
            "Row-level random splitting leaks users across train and test. The "
            "label here is mostly between-person variance, so a random split "
            "measures user identification rather than risk prediction, and it "
            "does not match deployment (new users have no labels at all). Use "
            "split_mode='user', or CVHealthDataModule for repeated grouped CV."
        ),
    }

    def __init__(
        self,
        split_mode: Literal["user", "longitudinal"] = "user",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        purge_hours: float = 0.0,
        swap_val_test: bool = False,
        symmetric_purge_diagnostic: bool = False,
    ) -> None:
        """Initializes the CohortSplitter.

        Args:
            split_mode: "user" (disjoint populations) or "longitudinal"
                (per-user temporal split).
            train_val_test_split: Fractions of each user's responses assigned
                to train, val, and test.
            random_state: Seed for the "user" mode's random partitioning.
            purge_hours: Only used by "longitudinal" mode. Width of the gap
                dropped around each user's train/val and val/test boundary,
                so that no retained sample's sampler window can reach across
                a split boundary into data used on the other side of it. Set
                this to at least the sampler's own lookback (its
                ``window_bounds`` span) — see ``lookback_hours_from_sampler``.
                0 (the default) reproduces the previous unpurged behaviour.
            swap_val_test: Only used by "longitudinal" mode. Diagnostic flag
                for isolating a role effect (val vs. test position) from a
                content effect (which chunk of a user's timeline it is). The
                two post-train chunks are cut and purged exactly as usual —
                this only relabels which chunk is returned as ``val_df`` and
                which as ``test_df``. The middle chunk (normally "val",
                sized by ``val_ratio``) is returned as test; the last chunk
                (normally "test", sized by ``test_ratio``) is returned as
                val. Because purging is positional, not label-based, the
                purge asymmetry flips too: the middle chunk (now "test") is
                purged against the last chunk's start, and the last chunk
                (now "val") is never purged, since nothing follows it. This
                is the mirror image of the unswapped case, not a bug — see
                the class docstring. Default False reproduces standard
                behaviour.
            symmetric_purge_diagnostic: Only used by "longitudinal" mode.
                A second diagnostic, distinct from ``swap_val_test``: isolates
                whether the middle vs. last post-train chunk differ in
                difficulty for a *content* reason, independent of purging.
                Normally only the middle chunk is purged (against the last
                chunk's start) — the last chunk never is, since nothing
                follows it, which confounds "this chunk is purged" with
                "this chunk is later" whenever the two chunks are compared.
                When True, purging instead drops rows within ``purge_hours``
                of each of the two post-train chunks' *own* start, applying
                the identical treatment to both regardless of position. If a
                score gap between the two chunks survives this symmetric
                purge, it reflects the chunks' content, not which one used to
                get purged; if it collapses, the original gap was purge-driven.
                Mutually exclusive with ``swap_val_test`` (raises if both are
                set — combining them doesn't correspond to a meaningful
                comparison). Default False reproduces standard behaviour.
        """
        self.split_mode = split_mode
        self.train_val_test_split = train_val_test_split
        self.random_state = random_state
        self.purge_hours = purge_hours
        self.swap_val_test = swap_val_test
        self.symmetric_purge_diagnostic = symmetric_purge_diagnostic
        if swap_val_test and symmetric_purge_diagnostic:
            raise ValueError(
                "swap_val_test and symmetric_purge_diagnostic are mutually "
                "exclusive diagnostics answering different questions — "
                "combining them doesn't correspond to a meaningful comparison."
            )

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Dispatches to the appropriate splitting strategy.

        Args:
            df: The master linked dataframe to split.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if self.split_mode in self.REMOVED_SPLIT_MODES:
            raise ValueError(
                f"split_mode='{self.split_mode}' is no longer supported. "
                f"{self.REMOVED_SPLIT_MODES[self.split_mode]}"
            )

        split_strategies = {
            "user": self._split_by_user,
            "longitudinal": self._split_longitudinally,
        }

        strategy_fn = split_strategies.get(self.split_mode)
        if not strategy_fn:
            raise ValueError(
                f"Unknown split_mode: {self.split_mode}. "
                f"Available modes: {list(split_strategies.keys())}"
            )

        return strategy_fn(df)

    def _split_by_user(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a split at the user level to ensure disjoint populations."""
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        unique_users = df["app_user_id"].unique()

        train_users, temp_users = train_test_split(
            unique_users,
            test_size=(1 - train_ratio),
            random_state=self.random_state,
        )

        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_users, test_users = train_test_split(
            temp_users,
            train_size=val_ratio_relative,
            random_state=self.random_state,
        )

        train_df = df[df["app_user_id"].isin(train_users)]
        val_df = df[df["app_user_id"].isin(val_users)]
        test_df = df[df["app_user_id"].isin(test_users)]
        return train_df, val_df, test_df

    def _split_longitudinally(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a temporal split within each individual user's history.

        Responses are first cut proportionally by chronological position, as
        before. If ``purge_hours`` is set, responses within that many hours of
        a train/val or val/test boundary are then dropped from the side
        closer to the boundary (see the class docstring for why). Purging
        never moves a response to a different side — it only removes samples
        whose sampler window would otherwise reach across the boundary — so a
        wide ``purge_hours`` relative to a user's response cadence can leave
        a user with fewer than the nominal proportional count in a split, or
        with none at all.

        The middle and last chunks are always cut and purged positionally
        first; ``swap_val_test`` only relabels which one is returned as
        ``val_df`` vs. ``test_df`` (see ``__init__``).
        """
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        purge = pd.Timedelta(hours=self.purge_hours) if self.purge_hours else pd.Timedelta(0)
        train_list, val_list, test_list = [], [], []

        for _, group in df.groupby("app_user_id"):
            group = group.sort_values("record_timestamp")
            n = len(group)

            if n < 3:
                # Not enough data to split 3 ways, default to training
                train_list.append(group)
                continue

            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            # Ensure at least one sample in each set where possible
            train_end = max(1, train_end)
            val_end = max(train_end + 1, val_end)

            train_part = group.iloc[:train_end]
            val_part = group.iloc[train_end:val_end]
            test_part = group.iloc[val_end:]

            if purge > pd.Timedelta(0):
                if self.symmetric_purge_diagnostic:
                    # Diagnostic only: purge each post-train chunk against its
                    # OWN start, not against the next chunk's start. Both
                    # chunks get identical treatment, so a boundary-adjacency
                    # asymmetry can't explain any score gap between them.
                    if not val_part.empty:
                        val_start = val_part["record_timestamp"].iloc[0]
                        val_part = val_part[
                            val_part["record_timestamp"] >= val_start + purge
                        ]
                    if not test_part.empty:
                        test_start = test_part["record_timestamp"].iloc[0]
                        test_part = test_part[
                            test_part["record_timestamp"] >= test_start + purge
                        ]
                else:
                    if not val_part.empty:
                        val_start = val_part["record_timestamp"].iloc[0]
                        train_part = train_part[
                            train_part["record_timestamp"] < val_start - purge
                        ]
                    if not test_part.empty:
                        test_start = test_part["record_timestamp"].iloc[0]
                        val_part = val_part[
                            val_part["record_timestamp"] < test_start - purge
                        ]

            train_list.append(train_part)
            val_list.append(val_part)
            test_list.append(test_part)

        train_df = pd.concat(train_list) if train_list else df.iloc[0:0]
        middle_df = pd.concat(val_list) if val_list else df.iloc[0:0]
        last_df = pd.concat(test_list) if test_list else df.iloc[0:0]

        if self.swap_val_test:
            # Middle chunk (purged, normally "val") -> returned as test.
            # Last chunk (unpurged, normally "test") -> returned as val.
            return train_df, last_df, middle_df
        return train_df, middle_df, last_df

    def split_walk_forward(
        self,
        df: pd.DataFrame,
        burn_in_responses: int,
        step_responses: int,
        val_responses: int = 0,
    ) -> List[WalkForwardFold]:
        """Per-user, purged, expanding-window walk-forward split.

        Answers "given a user's own history, can the model forecast their
        near-future risk?" — as distinct from ``split_mode="longitudinal"``'s
        single train/val/test cut, which only samples this question once per
        user. Here each user contributes multiple sequential forecast folds:
        fold *k*'s train set is everything before its test window (all of the
        user's own past, not a fixed-size trailing slice — the model is meant
        to accumulate history, mirroring how a deployed app would), and folds
        walk forward through the remainder of the user's responses.

        Purging is applied at *every* fold's train/test boundary (and, when
        ``val_responses`` is set, the train/val and val/test boundaries too)
        using the exact same logic as ``split_mode="longitudinal"`` — see the
        class docstring for why this is necessary (a sampler's lookback window
        can otherwise reach across a boundary into data assigned to the other
        side of it). ``self.purge_hours`` sets the width, same as elsewhere.

        A user contributes as many folds as their response count supports
        after ``burn_in_responses``; a user with fewer than
        ``burn_in_responses + val_responses + 1`` responses contributes none.
        This is expected, not an error — short-history users simply cannot
        be forecast-evaluated yet, and are reported as excluded rather than
        silently padded or dropped from the cohort entirely (they still
        appear in every fold's train set once they clear burn-in).

        Folds are aligned by position (fold 0, fold 1, ...) across users, not
        by calendar date — a user with more responses contributes more folds,
        a user with fewer contributes fewer, and the ``fold_index`` for a
        given user's Nth walk-forward step does not necessarily correspond to
        the same calendar period as another user's Nth step. Pool predictions
        across users within a fold index with this in mind: it is "the Nth
        forecast step for each user who has one", not "the same time window
        for everyone".

        Args:
            df: The master linked dataframe to split.
            burn_in_responses: Minimum number of a user's earliest responses
                reserved for the first fold's training set before any
                forecasting is evaluated for that user.
            step_responses: Number of responses each successive fold's test
                window covers, and by which the train window expands.
            val_responses: Number of responses each fold's validation window
                covers, sliced immediately after that fold's train window and
                immediately before its test window. Defaults to 0: no
                validation window at all. Every response that would otherwise
                go to val is absorbed into train instead (each fold is just
                train | purge | test), so no rows are set aside for early
                stopping / checkpoint selection — a caller running with no
                val split trains for its full configured epoch budget and
                evaluates the final-epoch weights.

        Returns:
            A list of ``WalkForwardFold`` namedtuples, ordered first by
            ``fold_index`` then by the order users appear in ``df``. Empty
            if no user clears burn-in. ``val_df`` is an empty (but correctly
            columned) DataFrame on every fold when ``val_responses == 0``.
        """
        if burn_in_responses < 1:
            raise ValueError(f"burn_in_responses must be >= 1; got {burn_in_responses}")
        if step_responses < 1:
            raise ValueError(f"step_responses must be >= 1; got {step_responses}")
        if val_responses < 0:
            raise ValueError(f"val_responses must be >= 0; got {val_responses}")

        purge = pd.Timedelta(hours=self.purge_hours) if self.purge_hours else pd.Timedelta(0)

        # Collect this user's folds as (train, val, test) triples, keyed by
        # fold_index, so folds can be regrouped and concatenated across users
        # by index afterward rather than by user.
        folds_by_index: dict = {}

        for _, group in df.groupby("app_user_id"):
            group = group.sort_values("record_timestamp")
            n = len(group)

            fold_index = 0
            train_end = burn_in_responses
            while True:
                val_end = train_end + val_responses
                test_end = val_end + step_responses
                if test_end > n:
                    break

                train_part = group.iloc[:train_end]
                val_part = group.iloc[train_end:val_end]  # empty when val_responses == 0
                test_part = group.iloc[val_end:test_end]

                if purge > pd.Timedelta(0):
                    if val_responses > 0:
                        if not val_part.empty:
                            val_start = val_part["record_timestamp"].iloc[0]
                            train_part = train_part[
                                train_part["record_timestamp"] < val_start - purge
                            ]
                        if not test_part.empty:
                            test_start = test_part["record_timestamp"].iloc[0]
                            val_part = val_part[
                                val_part["record_timestamp"] < test_start - purge
                            ]
                    else:
                        # No val window: purge train directly against test's
                        # own start, same boundary logic collapsed to one step.
                        if not test_part.empty:
                            test_start = test_part["record_timestamp"].iloc[0]
                            train_part = train_part[
                                train_part["record_timestamp"] < test_start - purge
                            ]

                folds_by_index.setdefault(fold_index, ([], [], []))
                folds_by_index[fold_index][0].append(train_part)
                folds_by_index[fold_index][1].append(val_part)
                folds_by_index[fold_index][2].append(test_part)

                fold_index += 1
                train_end = test_end  # expanding window: next fold's train includes this fold's test

        result: List[WalkForwardFold] = []
        for fold_index in sorted(folds_by_index.keys()):
            train_list, val_list, test_list = folds_by_index[fold_index]
            train_df = pd.concat(train_list) if train_list else df.iloc[0:0]
            val_df = pd.concat(val_list) if val_list else df.iloc[0:0]
            test_df = pd.concat(test_list) if test_list else df.iloc[0:0]
            result.append(WalkForwardFold(train_df, val_df, test_df, fold_index))

        return result

    def split_walk_forward_pct(
        self,
        df: pd.DataFrame,
        burn_in_pct: float,
        step_pct: float,
        val_pct: float = 0.0,
    ) -> List[WalkForwardFold]:
        """Per-user, purged, expanding-window walk-forward split, cut by percentage
        of each user's own response count rather than a fixed response count.

        Answers the same question as ``split_walk_forward`` ("given a user's own
        history, can the model forecast their near-future risk?"), with a
        different tradeoff: ``split_walk_forward`` sizes every fold's test
        window to the same fixed *count* of responses, so a user runs out of
        remaining data at a different fold than other users do, and later
        folds shed users one by one -- the last fold or two can end up with
        only a handful of users and a near-empty pooled test set. Here every
        fold's test window is instead a fixed *fraction* of that user's own
        total response count, so it scales with however much data a user has
        and every user who clears ``burn_in_pct`` contributes to every fold
        this method produces. Fold count is therefore uniform (bounded by
        ``(1.0 - burn_in_pct) / step_pct``, same math for every user) rather
        than emergent per-user the way ``split_walk_forward``'s is.

        Cuts are computed the same way ``split_mode="longitudinal"`` computes
        its single train/val/test cut (``int(n * fraction)``), just applied
        repeatedly instead of once. Purging at every fold's train/test
        boundary (and train/val, val/test when ``val_pct`` is set) uses the
        same logic as ``split_walk_forward`` -- see its docstring and the
        class docstring for why this matters.

        Folds are still aligned by position, not calendar date, with the same
        caveat as ``split_walk_forward``: fold *k* is "this user's Nth
        walk-forward step as a fraction of their own history," not a shared
        time window across users. What changes here is that pooling across
        users within a fold index is on much steadier footing, since every
        eligible user actually contributes to every fold rather than only
        the users with the most absolute data surviving into later folds.

        Args:
            df: The master linked dataframe to split.
            burn_in_pct: Fraction (0 < burn_in_pct < 1) of a user's earliest
                responses reserved for the first fold's training set before
                any forecasting is evaluated for that user.
            step_pct: Fraction of a user's total responses each successive
                fold's test window covers, and by which the train window
                expands. The loop stops once a fold's test window would
                extend past 100% of a user's responses.
            val_pct: Fraction of a user's total responses each fold's
                validation window covers. Defaults to 0.0: no validation
                window at all, same behavior and rationale as
                ``split_walk_forward``'s ``val_responses=0`` default -- see
                that method's docstring.

        Returns:
            A list of ``WalkForwardFold`` namedtuples, ordered first by
            ``fold_index`` then by the order users appear in ``df``. Empty
            if no user clears burn-in. ``val_df`` is an empty (but correctly
            columned) DataFrame on every fold when ``val_pct == 0.0``.
        """
        if not (0.0 < burn_in_pct < 1.0):
            raise ValueError(f"burn_in_pct must be in (0, 1); got {burn_in_pct}")
        if not (0.0 < step_pct < 1.0):
            raise ValueError(f"step_pct must be in (0, 1); got {step_pct}")
        if not (0.0 <= val_pct < 1.0):
            raise ValueError(f"val_pct must be in [0, 1); got {val_pct}")

        purge = pd.Timedelta(hours=self.purge_hours) if self.purge_hours else pd.Timedelta(0)

        folds_by_index: dict = {}

        for _, group in df.groupby("app_user_id"):
            group = group.sort_values("record_timestamp")
            n = len(group)

            fold_index = 0
            cum_pct = burn_in_pct
            while True:
                val_pct_end = cum_pct + val_pct
                test_pct_end = val_pct_end + step_pct
                if test_pct_end > 1.0:
                    break

                train_end = int(n * cum_pct)
                val_end = int(n * val_pct_end)
                test_end = int(n * test_pct_end)

                # A fold whose cut points round down to the same index (e.g.
                # a user with very few responses and a small step_pct) has no
                # test rows -- skip it for this user rather than emitting an
                # empty test fold that would look identical to "correctly
                # excluded by rounding" and "a real zero-width window".
                if test_end <= val_end:
                    break

                train_part = group.iloc[:train_end]
                val_part = group.iloc[train_end:val_end]  # empty when val_pct == 0
                test_part = group.iloc[val_end:test_end]

                if purge > pd.Timedelta(0):
                    if val_pct > 0.0:
                        if not val_part.empty:
                            val_start = val_part["record_timestamp"].iloc[0]
                            train_part = train_part[
                                train_part["record_timestamp"] < val_start - purge
                            ]
                        if not test_part.empty:
                            test_start = test_part["record_timestamp"].iloc[0]
                            val_part = val_part[
                                val_part["record_timestamp"] < test_start - purge
                            ]
                    else:
                        if not test_part.empty:
                            test_start = test_part["record_timestamp"].iloc[0]
                            train_part = train_part[
                                train_part["record_timestamp"] < test_start - purge
                            ]

                folds_by_index.setdefault(fold_index, ([], [], []))
                folds_by_index[fold_index][0].append(train_part)
                folds_by_index[fold_index][1].append(val_part)
                folds_by_index[fold_index][2].append(test_part)

                fold_index += 1
                cum_pct = test_pct_end  # expanding window: next fold's train includes this fold's test

        result: List[WalkForwardFold] = []
        for fold_index in sorted(folds_by_index.keys()):
            train_list, val_list, test_list = folds_by_index[fold_index]
            train_df = pd.concat(train_list) if train_list else df.iloc[0:0]
            val_df = pd.concat(val_list) if val_list else df.iloc[0:0]
            test_df = pd.concat(test_list) if test_list else df.iloc[0:0]
            result.append(WalkForwardFold(train_df, val_df, test_df, fold_index))

        return result

    def split_walk_forward_cyclic(
        self,
        df: pd.DataFrame,
        train_width_pct: float,
        step_pct: float,
    ) -> List[WalkForwardFold]:
        """Per-user, purged, FIXED-width walk-forward split that cycles through
        a user's full response history, wrapping past the end back to the start.

        Answers the same question as ``split_walk_forward``/``split_walk_forward_pct``
        ("given a user's own history, can the model forecast their near-future
        risk?"), but with a different notion of "history" than either: those two
        methods only ever grow train forward from a user's earliest response, so
        the earliest test folds are forecast from a short prefix of history and
        the latest ones from nearly all of it -- fold difficulty is confounded
        with how much training data was available. Here train is always the same
        *fixed width* (``train_width_pct`` of the user's total responses)
        immediately preceding test, and test tiles forward across the user's
        *entire* response range in ``step_pct``-sized windows -- including
        wrapping from the end of the response sequence back to the start, so a
        user's most recent responses can serve as "training history" for a test
        window sitting at the very beginning of their timeline. Every response
        in a user's history is tested on exactly once per full cycle, and every
        fold's train set is the same size, so folds are comparable to each other
        in a way the expanding-window methods' folds are not.

        This trades away the expanding-window methods' realism (a real deployment
        never trains on a user's future to predict their past) for uniform,
        directly-comparable folds and denser reuse of a short response history.
        Use ``split_walk_forward``/``split_walk_forward_pct`` when the expanding,
        past-only-training property matters (e.g. reporting a deployment-realistic
        forecast metric); use this when the goal is a stable per-fold estimate of
        within-user forecastability that isn't confounded by fold-to-fold history
        size, and training on cyclically-"future" data relative to a given test
        window is acceptable for that goal.

        Purging is applied at both adjacencies a wrapped fold can have with
        test. The near slice (train's tail, immediately before test's start)
        is purged against test's start, same timestamp-based logic as
        ``split_walk_forward``/``split_walk_forward_pct`` -- see the class
        docstring for why. When train wraps around the end of the sequence, it
        is physically two disjoint row-index slices that concatenate into one
        contiguous *cyclic*-time window, and the far slice (a wrapped train
        window's most-recent responses, sitting at the end of the user's raw
        timeline) is *also* adjacent to test -- to test's *end*, cyclically --
        so it is purged against that boundary the same way. This adjacency
        isn't a distant edge case: at parameters like
        ``train_width_pct + step_pct == 1.0`` (e.g. the 5-fold sweep default,
        0.80 + 0.20) the far slice's first row sits at the exact row test's
        last row precedes, zero gap, on every wrapping fold. Purging only the
        near side there would leave the model training on rows immediately
        adjacent to its own test window in time. Both purges only ever shrink
        their slice (drop rows falling inside the gap); they don't reflow train
        to backfill what purging removed, so a purged fold's effective train
        width can fall below ``train_width_pct`` -- consistent with how purge
        already behaves on the near side and in the other walk-forward
        methods.

        Args:
            df: The master linked dataframe to split.
            train_width_pct: Fraction (0 < train_width_pct < 1) of a user's
                total responses in every fold's train window. Fixed across
                folds, unlike the expanding-window methods.
            step_pct: Fraction of a user's total responses each fold's test
                window covers, and by which both train and test slide forward
                each fold. Folds tile the user's full 0..1 response range,
                wrapping past 1.0 back to 0.0, until test windows would start
                repeating (i.e. ``ceil(1.0 / step_pct)`` folds).

        Returns:
            A list of ``WalkForwardFold`` namedtuples (``val_df`` always empty
            -- this method has no validation-window concept, matching the
            ``val_responses=0`` / ``val_pct=0.0`` default elsewhere), ordered
            first by ``fold_index`` then by the order users appear in ``df``.
            Empty if no user has enough responses to form a full train window
            plus test window.
        """
        if not (0.0 < train_width_pct < 1.0):
            raise ValueError(f"train_width_pct must be in (0, 1); got {train_width_pct}")
        if not (0.0 < step_pct < 1.0):
            raise ValueError(f"step_pct must be in (0, 1); got {step_pct}")

        purge = pd.Timedelta(hours=self.purge_hours) if self.purge_hours else pd.Timedelta(0)
        n_folds = math.ceil(1.0 / step_pct)

        folds_by_index: dict = {}

        for _, group in df.groupby("app_user_id"):
            group = group.sort_values("record_timestamp")
            n = len(group)

            min_required = round(n * train_width_pct) + 1
            if n < min_required:
                continue  # not enough responses for even one fixed-width train window plus one test row

            for fold_index in range(n_folds):
                # round() before truncating: fold_index * step_pct accumulates
                # float error (e.g. 6 * 0.15 == 0.8999999999999999, not 0.9),
                # which int() would floor to one row short and cause a fold's
                # test window to overlap the previous fold's by one row.
                test_start_pct = min(round(fold_index * step_pct, 9), 1.0)
                test_end_pct = min(round(test_start_pct + step_pct, 9), 1.0)
                train_start_pct = round(test_start_pct - train_width_pct, 9)  # may be negative -> wraps

                test_start = round(n * test_start_pct)
                test_end = round(n * test_end_pct)
                if test_end <= test_start:
                    continue  # rounding collapsed this user's test window to nothing

                test_part = group.iloc[test_start:test_end]

                if train_start_pct < 0:
                    wrapped_start = round(n * (1.0 + train_start_pct))
                    far_part = group.iloc[wrapped_start:n]  # user's most-recent responses; not adjacent to test
                    near_part = group.iloc[0:test_start]  # adjacent to test's start; gets purged
                else:
                    train_start = round(n * train_start_pct)
                    far_part = group.iloc[0:0]
                    near_part = group.iloc[train_start:test_start]

                if purge > pd.Timedelta(0) and not test_part.empty:
                    if not near_part.empty:
                        test_start_ts = test_part["record_timestamp"].iloc[0]
                        near_part = near_part[near_part["record_timestamp"] < test_start_ts - purge]
                    if not far_part.empty:
                        # far_part is only ever populated on a wrapping fold, where it sits
                        # cyclically adjacent to test's *end* (see docstring) -- purge it
                        # against that boundary the same way near_part is purged against
                        # test's start.
                        test_end_ts = test_part["record_timestamp"].iloc[-1]
                        far_part = far_part[far_part["record_timestamp"] > test_end_ts + purge]

                train_part = pd.concat([far_part, near_part]) if len(far_part) else near_part
                if train_part.empty:
                    continue  # purge (or rounding) removed this fold's entire train window for this user

                folds_by_index.setdefault(fold_index, ([], []))
                folds_by_index[fold_index][0].append(train_part)
                folds_by_index[fold_index][1].append(test_part)

        empty_val = df.iloc[0:0]
        result: List[WalkForwardFold] = []
        for fold_index in sorted(folds_by_index.keys()):
            train_list, test_list = folds_by_index[fold_index]
            train_df = pd.concat(train_list) if train_list else df.iloc[0:0]
            test_df = pd.concat(test_list) if test_list else df.iloc[0:0]
            result.append(WalkForwardFold(train_df, empty_val, test_df, fold_index))

        return result
