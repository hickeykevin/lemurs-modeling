from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
from src.data.health_datamodule import HealthDataModule


class CVHealthDataModule(HealthDataModule):
    """A cross-validation DataModule for Multimodal longitudinal health data.

    This subclass overrides the standard HealthDataModule splitting logic
    to perform Group K-Fold cross validation on `app_user_id`.
    """

    def __init__(
        self,
        aggregator: LabelAggregator,
        sampler: TimeSampler,
        num_folds: int = 5,
        current_fold: int = 0,
        scaler: Optional[Any] = None,
        preprocessors: Optional[Dict[str, Any]] = None,
        modalities: List[str] = ["step"],
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "mean",
        use_prev_prediction: bool = False,
    ) -> None:
        # Initialize the base class, split_mode doesn't strictly matter as we override _split_data
        super().__init__(
            aggregator=aggregator,
            sampler=sampler,
            scaler=scaler,
            preprocessors=preprocessors,
            modalities=modalities,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_val_test_split=train_val_test_split,
            random_state=random_state,
            split_mode="user", 
            os_filter=os_filter,
            collapse_strategy=collapse_strategy,
            use_prev_prediction=use_prev_prediction,
        )
        # Re-save hyperparameters to capture num_folds and current_fold
        self.save_hyperparameters(logger=False)

    def get_num_folds(self) -> int:
        """Returns the number of folds, resolving -1 to the number of unique users."""
        if not hasattr(self, "master_df"):
            self.setup()
        return len(self.master_df['app_user_id'].unique())

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs Group K-Fold split at the user level for the current fold."""
        unique_users = df['app_user_id'].unique()

        num_folds = self.hparams.num_folds
        if num_folds == -1:
            num_folds = len(unique_users)

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.hparams.random_state)
        splits = list(kf.split(unique_users))

        # Get the train/test indices for the current fold
        train_idx, test_idx = splits[self.hparams.current_fold]
        train_users_full = unique_users[train_idx]
        test_users = unique_users[test_idx]

        # Further split the train users to extract a validation set for early stopping
        train_ratio, val_ratio, _ = self.hparams.train_val_test_split
        val_relative = val_ratio / (train_ratio + val_ratio)

        train_users, val_users = train_test_split(
            train_users_full,
            test_size=val_relative,
            random_state=self.hparams.random_state
        )

        train_df = df[df['app_user_id'].isin(train_users)]
        val_df = df[df['app_user_id'].isin(val_users)]
        test_df = df[df['app_user_id'].isin(test_users)]

        return train_df, val_df, test_df
