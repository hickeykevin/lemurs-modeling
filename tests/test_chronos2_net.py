import pytest
import torch

from src.models.components._sensor_channels import N_SENSOR_CHANNELS, SENSOR_CHANNELS, WEEK_HOURS
from src.models.components.chronos2_net import CHRONOS2_EMBED_DIM, Chronos2EncoderNet


def test_chronos2_encoder_net_constructs_without_chronos_forecasting():
    """Construction (and the HealthLitModule-facing interface) must not require the
    `chronos-forecasting` package -- only an actual forward pass does."""
    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    assert net.pipeline is None
    assert net.fc.in_features == CHRONOS2_EMBED_DIM

    net.init_demographics(demographics_dim=5)
    assert net.fc.in_features == CHRONOS2_EMBED_DIM + 5

    net.init_input_size(input_size=6)  # 2 modalities + 4 time-feature columns: fine
    with pytest.raises(ValueError):
        net.init_input_size(input_size=1)  # fewer columns than modalities: not fine


def test_chronos2_encoder_net_rejects_non_sequence_data():
    with pytest.raises(ValueError):
        Chronos2EncoderNet(modalities=["step"], use_sequence_data=False)


def test_chronos2_encoder_net_rejects_unfrozen_encoder():
    """Fine-tuning isn't supported yet -- see the module docstring's autograd caveat."""
    with pytest.raises(ValueError):
        Chronos2EncoderNet(modalities=["step"], freeze_encoder=False)


@pytest.mark.slow
def test_chronos2_encoder_net_forward_shape():
    """A real forward pass: downloads the checkpoint (network required, ~480MB, cached
    after the first run) and runs it -- CPU-capable, unlike the WBM/mamba_ssm net, so
    this runs anywhere rather than being skipped outside a GPU node with the extra
    installed. Marked `slow` (`make test` skips it) since it still does real inference."""
    pytest.importorskip("chronos", reason="only runs where the `chronos2` extra is installed")

    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    x = torch.randn(2, 16, 2)  # matches configs/data/sampler/rolling.yaml: 96h/6h bins
    logits = net(x)
    assert logits.shape == (2, 2)

    # A second forward pass reuses the already-loaded pipeline (no repeat download/load).
    logits2 = net(torch.randn(3, 16, 2))
    assert logits2.shape == (3, 2)


@pytest.mark.slow
def test_chronos2_encoder_net_forward_with_demographics():
    pytest.importorskip("chronos", reason="only runs where the `chronos2` extra is installed")

    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    net.init_demographics(demographics_dim=3)
    x = torch.randn(2, 16, 2)
    demographics = torch.randn(2, 3)
    logits = net(x, demographics=demographics)
    assert logits.shape == (2, 2)


def test_sensor_channels_shared_with_wbm():
    """Chronos-2 and WBM are pretrained/fine-tuned on the same channel vocabulary and
    weekly window -- confirms the two nets actually share it (see _sensor_channels.py)."""
    from src.models.components.wbm_net import SENSOR_CHANNELS as wbm_channels

    assert SENSOR_CHANNELS is wbm_channels
    assert N_SENSOR_CHANNELS == 19
    assert WEEK_HOURS == 168
