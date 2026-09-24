import pytest
import torch

from src.models.components.chronos_bolt_net import (
    BOLT_EMBED_DIMS,
    DEFAULT_BOLT_MODEL_ID,
    ChronosBoltEncoderNet,
)


def test_chronos_bolt_net_constructs_without_pipeline():
    """Construction must be lightweight and not download the model until forward is called."""
    net = ChronosBoltEncoderNet(modalities=["step", "calorie"], output_size=2)
    assert net.pipeline is None
    assert net.embed_dim == BOLT_EMBED_DIMS[DEFAULT_BOLT_MODEL_ID]
    assert net.fc.in_features == net.embed_dim

    net.init_demographics(demographics_dim=5)
    assert net.fc.in_features == net.embed_dim + 5

    net.init_input_size(input_size=6)  # 2 modalities + 4 time-feature columns: fine
    with pytest.raises(ValueError):
        net.init_input_size(input_size=1)  # fewer columns than modalities: raises


def test_chronos_bolt_net_concat_modality_pooling():
    """Verify concatenation across modalities expands representation dimension."""
    net = ChronosBoltEncoderNet(
        modalities=["step", "calorie", "distance"],
        modality_pooling="concat",
        output_size=2,
    )
    assert net.representation_dim == net.embed_dim * 3
    assert net.fc.in_features == net.embed_dim * 3


def test_chronos_bolt_net_rejects_non_sequence_data():
    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], use_sequence_data=False)


def test_chronos_bolt_net_invalid_pooling_raises():
    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], pooling="invalid")

    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], modality_pooling="invalid")


@pytest.mark.slow
def test_chronos_bolt_forward_pass_arbitrary_lengths():
    """Real forward pass verifying Chronos-Bolt handles arbitrary lengths T (no 168h padding required)."""
    pytest.importorskip("chronos", reason="only runs where the chronos2 extra is installed")

    net = ChronosBoltEncoderNet(
        modalities=["step", "calorie"],
        model_id="amazon/chronos-bolt-mini",
        output_size=2,
    )

    # Test with T = 16 (e.g. 6-hour bins over 96 hours)
    x16 = torch.randn(2, 16, 2)
    out16 = net(x16)
    assert out16.shape == (2, 2)

    # Test with T = 24 (e.g. 1-hour bins over 24 hours) - arbitrary length
    x24 = torch.randn(3, 24, 2)
    out24 = net(x24)
    assert out24.shape == (3, 2)

    # Test with trailing cyclic time features present (e.g. 2 modalities + 4 time features = 6 columns)
    x_with_time = torch.randn(2, 20, 6)
    out_time = net(x_with_time)
    assert out_time.shape == (2, 2)


@pytest.mark.slow
def test_chronos_bolt_forward_with_demographics():
    """Real forward pass verifying demographic concatenation."""
    pytest.importorskip("chronos", reason="only runs where the chronos2 extra is installed")

    net = ChronosBoltEncoderNet(
        modalities=["step", "calorie"],
        model_id="amazon/chronos-bolt-mini",
        output_size=2,
    )
    net.init_demographics(demographics_dim=4)

    x = torch.randn(2, 18, 2)
    demo = torch.randn(2, 4)
    out = net(x, demographics=demo)
    assert out.shape == (2, 2)


def test_chronos_bolt_dual_scaler_init():
    """Verify that init_input_size recognizes DualScaler channels (2*M) and updates concat pooling."""
    net = ChronosBoltEncoderNet(
        modalities=["step", "calorie"],
        modality_pooling="concat",
        output_size=2,
    )
    # Total cols = 2 modalities * 2 (dual) + 4 time features = 8 cols
    net.init_input_size(input_size=8, num_time_features=4)
    assert net.num_sensor_cols == 4  # 2 global + 2 subject
    assert net.representation_dim == net.embed_dim * 4
    assert net.fc.in_features == net.embed_dim * 4

    # Now with demographics
    net.init_demographics(demographics_dim=6)
    assert net.fc.in_features == (net.embed_dim * 4) + 6


@pytest.mark.slow
def test_chronos_bolt_forward_with_dual_scaler():
    """Real forward pass verifying DualScaler streams are encoded alongside demographics."""
    pytest.importorskip("chronos", reason="only runs where the chronos2 extra is installed")

    net = ChronosBoltEncoderNet(
        modalities=["step", "calorie"],
        modality_pooling="concat",
        model_id="amazon/chronos-bolt-mini",
        output_size=2,
    )
    # DualScaler: 2 global + 2 subject + 4 time features = 8 cols
    net.init_input_size(input_size=8, num_time_features=4)
    net.init_demographics(demographics_dim=3)

    x = torch.randn(2, 16, 8)
    demo = torch.randn(2, 3)
    out = net(x, demographics=demo)
    assert out.shape == (2, 2)

