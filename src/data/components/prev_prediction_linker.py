from typing import Tuple
import pandas as pd


class PrevPredictionLinker:
    """Precomputes chronological link indices between adjacent daily samples for each user."""

    @staticmethod
    def link(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Sorts the splits chronologically and computes the prev_sample_idx column.

        Args:
            train_df: Training dataframe.
            val_df: Validation dataframe.
            test_df: Test dataframe.

        Returns:
            Tuple of sorted and linked (train_df, val_df, test_df).
        """
        train_df = train_df.sort_values(["app_user_id", "record_timestamp"]).reset_index(drop=True)
        val_df = val_df.sort_values(["app_user_id", "record_timestamp"]).reset_index(drop=True)
        test_df = test_df.sort_values(["app_user_id", "record_timestamp"]).reset_index(drop=True)

        for df in [train_df, val_df, test_df]:
            prev_indices = []
            for idx in range(len(df)):
                if idx > 0 and df.iloc[idx - 1]["app_user_id"] == df.iloc[idx]["app_user_id"]:
                    prev_indices.append(idx - 1)
                else:
                    prev_indices.append(-1)
            df["prev_sample_idx"] = prev_indices

        return train_df, val_df, test_df
