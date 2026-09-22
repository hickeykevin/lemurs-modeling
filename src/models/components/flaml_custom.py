from typing import Any, Dict, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from flaml import AutoML, tune
from flaml.automl.model import SKLearnEstimator


class SVMPipeline(Pipeline):
    """Scikit-learn Pipeline combining StandardScaler and SVC with calibrated probabilities."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        random_state: int = 10242048,
        **kwargs: Any,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        svc = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=random_state,
        )
        super().__init__([("scaler", StandardScaler()), ("svc", svc)])

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None, **fit_params: Any) -> "SVMPipeline":
        """Fits pipeline, forwarding sample_weight to the SVC step."""
        if sample_weight is not None:
            fit_params["svc__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


class SVMPipelineEstimator(SKLearnEstimator):
    """FLAML custom learner for automated hyperparameter tuning of a scaled Support Vector Machine."""

    @classmethod
    def search_space(cls, **params: Any) -> Dict[str, Dict[str, Any]]:
        return {
            "C": {
                "domain": tune.loguniform(lower=0.01, upper=100.0),
                "init_value": 1.0,
            },
            "kernel": {
                "domain": tune.choice(["rbf", "linear"]),
                "init_value": "rbf",
            },
            "gamma": {
                "domain": tune.choice(["scale", "auto"]),
                "init_value": "scale",
            },
        }

    def __init__(self, task: str = "binary", **config: Any) -> None:
        super().__init__(task, **config)
        self.estimator_class = SVMPipeline


def register_custom_learners(automl: AutoML) -> None:
    """Registers custom estimators into a FLAML AutoML instance."""
    automl.add_learner("svm_pipeline", SVMPipelineEstimator)
