from typing import Tuple, Literal
import pandas as pd
from sklearn.model_selection import train_test_split


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
    ) -> None:
        self.split_mode = split_mode
        self.train_val_test_split = train_val_test_split
        self.random_state = random_state

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
        """Performs a temporal split within each individual user's history."""
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
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

            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])

        return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)
