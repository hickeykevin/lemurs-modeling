import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


MODALITY_CONFIG_CASES = [
    ("step", ["step"], ["step"]),
    ("calorie", ["calorie"], ["calorie"]),
    ("distance", ["distance"], ["distance"]),
    ("step_calorie", ["step", "calorie"], ["step", "calorie"]),
    ("step_distance", ["step", "distance"], ["step", "distance"]),
    ("calorie_distance", ["calorie", "distance"], ["calorie", "distance"]),
    ("step_calorie_distance", ["step", "calorie", "distance"], ["step", "calorie", "distance"]),
    ("all", ["step", "calorie", "distance"], ["step", "calorie", "distance"]),
]


@pytest.fixture(autouse=True)
def clean_hydra():
    """Ensure GlobalHydra is cleaned before and after tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("config_name,expected_modalities,expected_preprocessors", MODALITY_CONFIG_CASES)
def test_modality_configs_compose(config_name, expected_modalities, expected_preprocessors):
    """Test that each data/modalities config populates data.modalities and data.preprocessors correctly."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=[f"data/modalities={config_name}"])
        assert isinstance(cfg, DictConfig)
        assert list(cfg.data.modalities) == expected_modalities
        assert list(cfg.data.preprocessors.keys()) == expected_preprocessors


@pytest.mark.parametrize("data_name", ["default", "single_split", "user_cv", "walk_forward_expanding", "walk_forward_cyclic"])
@pytest.mark.parametrize("modality_choice", ["step", "distance", "step_calorie_distance"])
def test_modality_override_on_data_schemes(data_name, modality_choice):
    """Test overriding data/modalities across all data configs."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"data={data_name}", f"data/modalities={modality_choice}"],
        )
        assert isinstance(cfg, DictConfig)
        expected = MODALITY_CONFIG_CASES[[c[0] for c in MODALITY_CONFIG_CASES].index(modality_choice)]
        assert list(cfg.data.modalities) == expected[1]
        assert list(cfg.data.preprocessors.keys()) == expected[2]


def test_modality_len_interpolations():
    """Test that len:${data.modalities} resolves correctly for model input_size."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train",
            overrides=["model=default", "data/modalities=step_calorie_distance"],
        )
        # In LSTM net, input_size is ${len:${data.modalities}}
        assert cfg.model.net.input_size == 3

        cfg2 = compose(
            config_name="train",
            overrides=["model=default", "data/modalities=step"],
        )
        assert cfg2.model.net.input_size == 1
