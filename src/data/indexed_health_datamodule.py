from typing import Optional

from src.data.components.health_dataset import HealthDataset
from src.data.health_datamodule import HealthDataModule


class IndexedHealthDataModule(HealthDataModule):
    """``HealthDataModule`` with ``data_val``/``data_test`` rebuilt to carry
    their sample index, so predictions can be joined back to source rows.

    Exists so a single-split evaluation (e.g. ``split_mode="longitudinal"``)
    can use ``PredictionCollectorCallback`` and the same pooled, user-cluster
    bootstrap metrics ``wf_cv_train.py`` computes for its CV schemes --
    without which a longitudinal run's only CI comes from
    ``ClassificationMetricsCallback``'s row-level ``torchmetrics.BootStrapper``,
    which (like every row-level bootstrap discussed elsewhere in this
    pipeline -- see ``CohortSplitter``'s class docstring) understates
    uncertainty by treating each of one user's many responses as an
    independent draw, rather than resampling whole participants.

    This is exactly the rebuild step ``WalkForwardHealthDataModule.setup()``
    performs on top of its own fold-selecting ``_split_data`` override --
    see that class's docstring for the full rationale. Here there is no fold
    logic at all: every ``HealthDataModule`` split_mode ("user" or
    "longitudinal") is supported unchanged, and this class only adds the
    return_index=True rebuild on top of whichever one the base class ran.
    """

    def setup(self, stage: Optional[str] = None) -> None:
        """Runs the base class's normal setup, then rebuilds ``data_val``/
        ``data_test`` with ``return_index=True``, reusing the scaler/
        demographics state the base class already fit -- identical in
        mechanics to ``WalkForwardHealthDataModule.setup()``, minus the
        fold selection.
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
