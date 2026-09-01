import pytest
import torch

from src.models.components.wbm_net import (
    N_SENSOR_CHANNELS,
    SENSOR_CHANNELS,
    WBMEncoderNet,
    assemble_weekly_tensor,
)


def test_assemble_weekly_tensor_shape_and_missingness():
    # 96h lookback / 6h bins = 16 bins, matching configs/data/sampler/rolling.yaml.
    x_sensor = torch.randn(4, 16, 2)  # modalities: ["step", "calorie"]
    out = assemble_weekly_tensor(
        x_sensor,
        modalities=["step", "calorie"],
        modality_channel_map={"step": "iphone_steps", "calorie": "watch_energy"},
        resample_freq_hours=6.0,
    )
    assert out.shape == (4, 168, 2 * N_SENSOR_CHANNELS)

    values, missing = out[..., :N_SENSOR_CHANNELS], out[..., N_SENSOR_CHANNELS:]
    steps_idx = SENSOR_CHANNELS.index("iphone_steps")
    energy_idx = SENSOR_CHANNELS.index("watch_energy")
    distance_idx = SENSOR_CHANNELS.index("iphone_distance")

    # 96 observed hours were left-padded to 168: the oldest 72 hours are missing
    # everywhere, including on mapped channels.
    assert torch.all(missing[:, :72, :] == 1.0)
    # Mapped channels are observed for the remaining (most recent) 96 hours.
    assert torch.all(missing[:, 72:, steps_idx] == 0.0)
    assert torch.all(missing[:, 72:, energy_idx] == 0.0)
    # An unmapped channel stays fully "missing" everywhere.
    assert torch.all(missing[:, :, distance_idx] == 1.0)
    assert torch.all(values[:, :, distance_idx] == 0.0)


def test_assemble_weekly_tensor_repeats_bins_across_hours():
    x_sensor = torch.zeros(1, 4, 1)
    x_sensor[0, 2, 0] = 5.0  # third 6-hour bin
    out = assemble_weekly_tensor(
        x_sensor,
        modalities=["step"],
        modality_channel_map={"step": "iphone_steps"},
        resample_freq_hours=6.0,
    )
    steps_idx = SENSOR_CHANNELS.index("iphone_steps")
    values = out[..., :N_SENSOR_CHANNELS]

    # 4 bins * 6h = 24 observed hours, left-padded by 144h to reach 168.
    # Bin index 2 (0-indexed) lands at hours [144+12 : 144+18).
    expanded = values[0, :, steps_idx]
    assert torch.all(expanded[144 + 12 : 144 + 18] == 5.0)
    assert torch.all(expanded[: 144 + 12] == 0.0)
    assert torch.all(expanded[144 + 18 :] == 0.0)


def test_assemble_weekly_tensor_truncates_long_lookback():
    # 30 bins * 8h = 240 observed hours > 168 -> keep only the most recent week.
    x_sensor = torch.arange(30, dtype=torch.float32).view(1, 30, 1)
    out = assemble_weekly_tensor(
        x_sensor,
        modalities=["step"],
        modality_channel_map={"step": "iphone_steps"},
        resample_freq_hours=8.0,
    )
    assert out.shape == (1, 168, 2 * N_SENSOR_CHANNELS)
    missing = out[..., N_SENSOR_CHANNELS:]
    steps_idx = SENSOR_CHANNELS.index("iphone_steps")
    assert torch.all(missing[:, :, steps_idx] == 0.0)  # nothing left to pad as missing


def test_assemble_weekly_tensor_rejects_fractional_bin_width():
    x_sensor = torch.zeros(1, 4, 1)
    with pytest.raises(ValueError):
        assemble_weekly_tensor(
            x_sensor, modalities=["step"], modality_channel_map={"step": "iphone_steps"}, resample_freq_hours=1.5
        )


def test_wbm_encoder_net_constructs_without_mamba_ssm():
    """Construction (and the HealthLitModule-facing interface) must not require the
    CUDA-only mamba_ssm package -- only an actual forward pass does."""
    net = WBMEncoderNet(modalities=["step", "calorie"], output_size=2)
    assert net.encoder is None
    assert net.fc.in_features == net.embed_dim

    net.init_demographics(demographics_dim=5)
    assert net.fc.in_features == net.embed_dim + 5

    net.init_input_size(input_size=6)  # 2 modalities + 4 time-feature columns: fine
    with pytest.raises(ValueError):
        net.init_input_size(input_size=1)  # fewer columns than modalities: not fine


def test_wbm_encoder_net_rejects_non_sequence_data():
    with pytest.raises(ValueError):
        WBMEncoderNet(modalities=["step"], use_sequence_data=False)


def test_wbm_encoder_net_forward_requires_mamba_ssm():
    pytest.importorskip("mamba_ssm", reason="only runs where the `wbm` extra is installed")
    # A real forward pass additionally needs network access to fetch the checkpoint and
    # a CUDA device to run it -- out of scope for CI; this just documents the contract.
    net = WBMEncoderNet(modalities=["step", "calorie"], output_size=2)
    x = torch.randn(2, 16, 2)
    logits = net(x)
    assert logits.shape == (2, 2)
