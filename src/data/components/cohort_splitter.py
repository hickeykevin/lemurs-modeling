from typing import Tuple, Literal
import pandas as pd
from sklearn.model_selection import train_test_split


class CohortSplitter:
    """Handles splitting the cohort dataset into train, validation, and test sets.

    Supports three strategies:
        - "random": Row-level random shuffle.
        - "user": Split by user ID (disjoint populations).
        - "longitudinal": Temporal split per user (predict future from past).
    """

    def __init__(
        self,
        split_mode: Literal["random", "user", "longitudinal"] = "random",
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
        split_strategies = {
            "random": self._split_random,
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

    def _split_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a standard row-level random shuffle split."""
        train_ratio, val_ratio, test_ratio = self.train_val_test_split

        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            random_state=self.random_state,
        )

        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_relative,
            random_state=self.random_state,
        )
        return train_df, val_df, test_df

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
