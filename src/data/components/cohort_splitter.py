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
        val_responses: int,
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

        Purging is applied at *every* fold's train/val and val/test boundary
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
                immediately before its test window.

        Returns:
            A list of ``WalkForwardFold`` namedtuples, ordered first by
            ``fold_index`` then by the order users appear in ``df``. Empty
            if no user clears burn-in.
        """
        if burn_in_responses < 1:
            raise ValueError(f"burn_in_responses must be >= 1; got {burn_in_responses}")
        if step_responses < 1:
            raise ValueError(f"step_responses must be >= 1; got {step_responses}")
        if val_responses < 1:
            raise ValueError(f"val_responses must be >= 1; got {val_responses}")

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
                val_part = group.iloc[train_end:val_end]
                test_part = group.iloc[val_end:test_end]

                if purge > pd.Timedelta(0):
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
