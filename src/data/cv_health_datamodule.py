import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.components.label_aggregators import LabelAggregator
from src.data.components.samplers import TimeSampler
from src.data.health_datamodule import HealthDataModule


def _get_default_high_risk_user_ids() -> List[int]:
    """Loads default high-risk participant IDs from HIGH_RISK_USER_IDS in .env."""
    env_val = os.getenv("HIGH_RISK_USER_IDS", "")
    if not env_val:
        return []
    ids = []
    for item in env_val.split(","):
        item_str = item.strip()
        if item_str:
            try:
                ids.append(int(item_str))
            except ValueError:
                ids.append(item_str)
    return ids


class CVHealthDataModule(HealthDataModule):
    """Repeated, stratified, user-grouped cross-validation over the cohort.

    Every split is grouped on ``app_user_id`` so no participant appears on both
    sides — the only arrangement that matches deployment, where a new user of
    the app never takes surveys and so has no labels the model could have seen.

    Two details matter at this cohort size (31 users with sensor coverage, of
    which roughly a dozen are ever positive):

    **Stratification.** Positives are concentrated in a handful of participants,
    so plain ``KFold`` over users routinely produces a fold with no positive
    test user at all, making AUROC undefined or wild. ``StratifiedGroupKFold``
    balances the label distribution across folds while keeping users intact.

    **Repeats.** A single 5-fold pass over ~31 users is dominated by which
    users happened to land together. Repeating the whole procedure with
    different shuffles and reporting the spread turns that noise into a stated
    uncertainty rather than a hidden one. Fold-to-fold variance is expected to
    be the largest term in any result this study reports.

    The validation set is carved from the training users with the same grouped,
    stratified procedure, so early stopping never sees a test user either.
    """

    def __init__(
        self,
        aggregator: LabelAggregator,
        sampler: TimeSampler,
        num_folds: int = 5,
        current_fold: int = 0,
        num_repeats: int = 20,
        current_repeat: int = 0,
        scaler: Optional[Any] = None,
        preprocessors: Optional[Dict[str, Any]] = None,
        modalities: List[str] = ["step"],
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        os_filter: Optional[Literal["ios", "android", "both"]] = "both",
        collapse_strategy: str = "none",
        use_prev_prediction: bool = False,
        use_demographics: bool = True,
        use_sleep: bool = False,
        enrollment_lead_days: float = 7.0,
        enrollment_trail_days: float = 1.0,
        require_sensor_data: bool = True,
        use_survey_context: bool = True,
        exclude_user_ids: Optional[List[int]] = None,
        high_risk_user_ids: Optional[List[int]] = None,
        prebuilt_cohort: Optional[Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]] = None,
    ) -> None:
        """Initializes the CVHealthDataModule."""
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
            use_demographics=use_demographics,
            use_sleep=use_sleep,
            enrollment_lead_days=enrollment_lead_days,
            enrollment_trail_days=enrollment_trail_days,
            require_sensor_data=require_sensor_data,
            use_survey_context=use_survey_context,
            exclude_user_ids=exclude_user_ids,
            prebuilt_cohort=prebuilt_cohort,
        )
        if high_risk_user_ids is None:
            high_risk_user_ids = _get_default_high_risk_user_ids()
        self.hparams.high_risk_user_ids = high_risk_user_ids
        # Re-save hyperparameters to capture the CV-specific fields
        self.save_hyperparameters(logger=False)

    def get_num_folds(self) -> int:
        """Returns the fold count, resolving -1 to leave-one-user-out."""
        if not hasattr(self, "master_df"):
            self.setup()
        return len(self.master_df["app_user_id"].unique())

    def _repeat_seed(self) -> int:
        """Derives this repeat's shuffle seed from the base random_state."""
        base = self.hparams.random_state
        base = 0 if base is None else int(base)
        return base + 9973 * int(self.hparams.current_repeat)

    @classmethod
    def _user_level_strata(
        cls, df: pd.DataFrame, high_risk_user_ids: Optional[List[int]] = None
    ) -> pd.Series:
        """Labels each row by whether its user is in the high-risk set.

        Stratifying on whether a participant is in the high-risk set balances
        the risk label distribution across folds while keeping users intact.
        Falls back to dynamic 'ever positive' survey response logic if
        high_risk_user_ids is explicitly set to an empty list or if no users
        in df match the high-risk ID set (e.g. synthetic test cohorts).
        """
        if high_risk_user_ids is None:
            high_risk_user_ids = _get_default_high_risk_user_ids()

        if high_risk_user_ids:
            high_risk_set = set()
            for u in high_risk_user_ids:
                high_risk_set.add(u)
                high_risk_set.add(str(u))
                try:
                    val = int(u)
                    high_risk_set.add(val)
                    if val > 10000:
                        high_risk_set.add(val % 10000)
                except (ValueError, TypeError):
                    pass

            strata = df["app_user_id"].apply(lambda u: u in high_risk_set or str(u) in high_risk_set).astype(int)
            if strata.sum() > 0:
                return strata

        ever_positive = df.groupby("app_user_id")["answer"].transform(
            lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).max() > 0)
        )
        return ever_positive.astype(int)

    def _grouped_split(
        self, df: pd.DataFrame, n_splits: int, fold: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (train_users, held_out_users) for one stratified grouped fold."""
        groups = df["app_user_id"].values
        high_risk = getattr(self.hparams, "high_risk_user_ids", None)
        strata = self._user_level_strata(df, high_risk_user_ids=high_risk).values

        n_users = len(np.unique(groups))
        n_splits = max(2, min(n_splits, n_users))

        # StratifiedGroupKFold needs at least one group per class per fold; if
        # one stratum is too small to spread, fall back to unstratified grouping.
        class_counts = pd.DataFrame({"g": groups, "s": strata}).drop_duplicates()["s"].value_counts()
        if len(class_counts) < 2 or class_counts.min() < n_splits:
            rng = np.random.RandomState(seed)
            users = np.unique(groups)
            rng.shuffle(users)
            chunks = np.array_split(users, n_splits)
            held_out = chunks[fold % n_splits]
            return np.setdiff1d(users, held_out), held_out

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(df)), strata, groups))
        train_idx, test_idx = splits[fold % n_splits]
        return (
            np.unique(groups[train_idx]),
            np.unique(groups[test_idx]),
        )

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits into train/val/test for the current (repeat, fold) pair."""
        seed = self._repeat_seed()

        num_folds = self.hparams.num_folds
        if num_folds == -1:
            num_folds = len(df["app_user_id"].unique())

        train_users_full, test_users = self._grouped_split(
            df, num_folds, self.hparams.current_fold, seed
        )

        # Carve validation from the training users using the same procedure, so
        # early stopping is also judged on unseen people.
        train_pool = df[df["app_user_id"].isin(train_users_full)]
        train_ratio, val_ratio, _ = self.hparams.train_val_test_split
        val_share = val_ratio / max(train_ratio + val_ratio, 1e-9)
        n_val_splits = max(2, int(round(1.0 / max(val_share, 1e-9))))

        train_users, val_users = self._grouped_split(train_pool, n_val_splits, 0, seed + 1)

        train_df = df[df["app_user_id"].isin(train_users)]
        val_df = df[df["app_user_id"].isin(val_users)]
        test_df = df[df["app_user_id"].isin(test_users)]

        return train_df, val_df, test_df
