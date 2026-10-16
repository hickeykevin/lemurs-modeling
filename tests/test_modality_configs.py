import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def clean_hydra():
    """Ensure GlobalHydra is cleaned before and after tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_default_modalities_are_step_and_distance():
    """The paper's runs use step + distance, with one preprocessor per modality."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train")
        assert list(cfg.data.modalities) == ["step", "distance"]
        assert list(cfg.data.preprocessors.keys()) == ["step", "distance"]


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_sequence_model_input_size_follows_modalities(model):
    """len:${data.modalities} resolves to the number of sensor streams."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=[f"model={model}"])
        assert cfg.model.net.input_size == 2
