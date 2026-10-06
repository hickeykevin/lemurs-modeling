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


class TabPFNModel:
    """scikit-learn style wrapper around Prior Labs' TabPFN classifier (default TabPFN-3.5).

    TabPFN is a pretrained tabular foundation model: ``fit`` stores the training set as
    context and ``predict_proba`` runs one forward pass, with no gradient training. It has no
    use for per-sample weights, so ``sample_weight`` (FLAML passes it under
    ``auto_class_weights``) is ignored; AUROC is rank-based, so class balancing would not
    change it. The ``tabpfn`` package is imported only when fitting (``uv sync --extra
    tabpfn``), and downloading the weights needs a one-time licence acceptance (the
    ``TABPFN_TOKEN`` environment variable; see ``tabpfn.errors.TabPFNLicenseError``).
    """

    def __init__(
        self,
        model_version: str = "v3.5",
        n_estimators: Any = "auto",
        device: str = "auto",
        random_state: int = 0,
        finetune: bool = False,
        finetune_epochs: int = 30,
        finetune_lr: float = 1e-5,
        **kwargs: Any,  # FLAML may pass n_jobs etc.; TabPFN manages its own parallelism
    ) -> None:
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.finetune = finetune
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"model_version": self.model_version, "n_estimators": self.n_estimators,
                "device": self.device, "random_state": self.random_state, "finetune": self.finetune,
                "finetune_epochs": self.finetune_epochs, "finetune_lr": self.finetune_lr}

    def set_params(self, **params: Any) -> "TabPFNModel":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _make_classifier(self):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if self.finetune:
            # Prior Labs' own fine-tuning loop: AdamW, early stopping on a 10% split of the
            # training data, gradient clipping, lr warmup/schedule; no checkpoints written.
            import torch
            from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

            device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            return FinetunedTabPFNClassifier(
                device=device, epochs=self.finetune_epochs, learning_rate=self.finetune_lr,
                random_state=self.random_state, model_version=ModelVersion(self.model_version),
            )
        return TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version),
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None, **fit_params: Any) -> "TabPFNModel":
        self.model_ = self._make_classifier()
        self.model_.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        # FinetunedTabPFNClassifier does not expose classes_; fall back to the labels' classes.
        self.classes_ = getattr(self.model_, "classes_", None)
        if self.classes_ is None:
            self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(np.asarray(X, dtype=np.float32))


class TabPFNEstimator(SKLearnEstimator):
    """FLAML learner for TabPFN. TabPFN is used as released, with nothing to tune, so the
    search space is a single point and FLAML evaluates it once."""

    @classmethod
    def search_space(cls, **params: Any) -> Dict[str, Dict[str, Any]]:
        return {"model_version": {"domain": tune.choice(["v3.5"]), "init_value": "v3.5"}}

    def __init__(self, task: str = "binary", **config: Any) -> None:
        super().__init__(task, **config)
        self.estimator_class = TabPFNModel


class TabPFNFinetunedEstimator(TabPFNEstimator):
    """FLAML learner for TabPFN fine-tuned on the training fold (Prior Labs' fine-tuning loop)."""

    @classmethod
    def search_space(cls, **params: Any) -> Dict[str, Dict[str, Any]]:
        return {"finetune": {"domain": tune.choice([True]), "init_value": True}}


def register_custom_learners(automl: AutoML) -> None:
    """Registers custom estimators into a FLAML AutoML instance."""
    automl.add_learner("svm_pipeline", SVMPipelineEstimator)
    automl.add_learner("tabpfn", TabPFNEstimator)
    automl.add_learner("tabpfn_ft", TabPFNFinetunedEstimator)
