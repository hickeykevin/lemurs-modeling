import pytest
import torch

from src.models.components._sensor_channels import N_SENSOR_CHANNELS, SENSOR_CHANNELS, WEEK_HOURS
from src.models.components.chronos2_net import (
    CHRONOS2_EMBED_DIM,
    INPUT_STAT_NAMES,
    Chronos2EncoderNet,
    predict_last_latent,
)

LEGACY_OPENMHC = dict(
    token_pooling="last", channel_pooling="mean", drop_unmapped_channels=False, append_input_stats=False
)


def test_chronos2_encoder_net_constructs_without_chronos_forecasting():
    """Construction (and the HealthLitModule-facing interface) must not require the
    `chronos-forecasting` package -- only an actual forward pass does."""
    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    assert net.pipeline is None
    # Default: one embedding per mapped channel (concat) + 5 input stats per modality.
    expected = 2 * CHRONOS2_EMBED_DIM + 2 * len(INPUT_STAT_NAMES)
    assert net.fc.in_features == expected

    net.init_demographics(demographics_dim=5)
    assert net.fc.in_features == expected + 5

    legacy = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2, **LEGACY_OPENMHC)
    assert legacy.fc.in_features == CHRONOS2_EMBED_DIM

    with pytest.warns(UserWarning):
        net.init_input_size(input_size=6)  # layout unknown without the time-feature count
    assert net.subject_cols is None and net.time_cols is None
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


def test_chronos2_rejects_unknown_pooling_modes():
    for bad in (dict(token_pooling="max"), dict(layer_pooling="first"), dict(channel_pooling="sum")):
        with pytest.raises(ValueError):
            Chronos2EncoderNet(modalities=["step"], **bad)


def test_chronos2_channel_selection_follows_modality_order():
    net = Chronos2EncoderNet(modalities=["calorie", "step"])
    assert net.channel_idx == [SENSOR_CHANNELS.index("watch_energy"), SENSOR_CHANNELS.index("iphone_steps")]
    assert Chronos2EncoderNet(modalities=["step"], drop_unmapped_channels=False).channel_idx == list(
        range(N_SENSOR_CHANNELS)
    )


def test_chronos2_input_stats():
    x = torch.tensor([[[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]])  # [B=1, T=3, M=2]
    stats = Chronos2EncoderNet.input_stats(x).reshape(1, 2, len(INPUT_STAT_NAMES))
    torch.testing.assert_close(stats[0, 0], torch.tensor([2.0, (2 / 3) ** 0.5, 1.0, 3.0, 2.0]))
    torch.testing.assert_close(stats[0, 1], torch.tensor([20.0, (200 / 3) ** 0.5, 10.0, 30.0, 30.0]))


def test_chronos2_observed_patch_mask_left_pads_like_chronos():
    from types import SimpleNamespace

    net = Chronos2EncoderNet(modalities=["step"])
    net.pipeline = SimpleNamespace(model=SimpleNamespace(chronos_config=SimpleNamespace(input_patch_size=4)))
    ctx = torch.full((1, 10), float("nan"))
    ctx[0, 7:] = 1.0  # 10 steps -> left-padded by 2 to 3 patches of 4; only the last patch is observed
    mask = net._observed_patch_mask(ctx, num_patches=3)
    assert mask.tolist() == [[False, False, True]]


def test_chronos2_embedding_cache_skips_seen_rows(monkeypatch):
    from src.models.components import chronos2_net

    monkeypatch.setattr(chronos2_net, "_EMBED_CACHE", {})
    net = Chronos2EncoderNet(modalities=["step"], append_input_stats=False)
    calls = []

    def fake_encode(x_sensor):
        calls.append(x_sensor.shape[0])
        return x_sensor.sum(dim=(1, 2)).unsqueeze(-1).repeat(1, net.embed_dim)

    monkeypatch.setattr(net, "_encode", fake_encode)
    a, b = torch.randn(1, 4, 1), torch.randn(1, 4, 1)
    first = net._embed(torch.cat([a, b]))
    second = net._embed(torch.cat([b, a, b]))  # all rows seen before
    assert calls == [2]
    torch.testing.assert_close(second[1], first[0])
    torch.testing.assert_close(second[0], first[1])


@pytest.mark.slow
@pytest.mark.parametrize("token_pooling", ["mean", "reg", "last"])
@pytest.mark.parametrize("layer_pooling", ["last", "mean"])
@pytest.mark.parametrize("channel_pooling", ["concat", "mean"])
def test_chronos2_forward_all_pooling_modes(token_pooling, layer_pooling, channel_pooling):
    pytest.importorskip("chronos", reason="only runs where the `chronos2` extra is installed")

    net = Chronos2EncoderNet(
        modalities=["step", "calorie", "distance"],
        output_size=2,
        resample_freq_hours=4.0,
        token_pooling=token_pooling,
        layer_pooling=layer_pooling,
        channel_pooling=channel_pooling,
        cache_embeddings=False,
    )
    x = torch.randn(4, 30, 3)  # 120h lookback at 4h bins
    net._ensure_pipeline(x.device)
    emb = net._embed(x)
    assert emb.shape == (4, net.embed_dim)
    assert torch.isfinite(emb).all()
    assert net(x).shape == (4, 2)
    # Samples must be encoded independently: a batch row's embedding does not depend
    # on what else is in the batch (group ids keep samples apart).
    torch.testing.assert_close(net._embed(x[:1]), emb[:1], rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_chronos2_legacy_config_matches_openmhc_extraction():
    """With OpenMHC's settings, the direct `encode` path must reproduce the original
    numpy round-trip through `Chronos2Dataset` (`predict_last_latent`)."""
    pytest.importorskip("chronos", reason="only runs where the `chronos2` extra is installed")
    from src.models.components._sensor_channels import resample_and_pad_to_week

    torch.manual_seed(0)
    net = Chronos2EncoderNet(modalities=["step", "calorie"], cache_embeddings=False, **LEGACY_OPENMHC)
    x = torch.randn(3, 16, 2).abs()  # 96h at 6h bins
    net._ensure_pipeline(x.device)
    ours = net._embed(x)

    values, observed = resample_and_pad_to_week(x, net.modalities, net.modality_channel_map, 6.0)
    x_chw = torch.where(observed, values, torch.full_like(values, float("nan"))).permute(0, 2, 1)
    with torch.no_grad():
        latents = predict_last_latent(
            net.pipeline, x_chw.numpy(), prediction_length=net.prediction_length,
            batch_size=3, context_length=WEEK_HOURS,
        )
    reference = torch.stack(latents).mean(dim=1)
    torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-4)


def test_chronos2_init_input_size_finds_subject_and_time_columns():
    """DualScaler layout: [global sensors | subject sensors | time features]. The head
    gets stats of both sensor blocks plus the last bin's time features."""
    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    net.init_input_size(input_size=2 + 2 + 4, num_time_features=4)
    assert net.subject_cols == slice(2, 4) and net.time_cols == slice(4, 8)
    expected = 2 * CHRONOS2_EMBED_DIM + 4 * len(INPUT_STAT_NAMES) + 4
    assert net.fc.in_features == expected
    net.init_demographics(demographics_dim=1)
    assert net.fc.in_features == expected + 1

    plain = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    plain.init_input_size(input_size=2, num_time_features=0)  # single scaler, no time features
    assert plain.subject_cols is None and plain.time_cols is None
    assert plain.fc.in_features == 2 * CHRONOS2_EMBED_DIM + 2 * len(INPUT_STAT_NAMES)

    with pytest.warns(UserWarning):
        net.init_input_size(input_size=7, num_time_features=4)  # 3 sensor cols for 2 modalities
    assert net.subject_cols is None and net.time_cols is None


@pytest.mark.slow
def test_chronos2_forward_with_dual_scaler_layout():
    pytest.importorskip("chronos", reason="only runs where the `chronos2` extra is installed")

    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    net.init_input_size(input_size=8, num_time_features=4)
    net.init_demographics(demographics_dim=1)
    logits = net(torch.randn(3, 12, 8), demographics=torch.randn(3, 1))
    assert logits.shape == (3, 2)
