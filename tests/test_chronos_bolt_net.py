"""ChronosBoltEncoderNet: pooling, layout, caching, fine-tuning, and a real-weights check.

The fast tests use a stand-in with Chronos-Bolt's ``model.encode`` interface: the series is
standardised, left-padded with NaN to whole 16-step patches, each patch embedded by a Linear, and
a [REG] token appended; ``encode`` returns (hidden, loc_scale, input_embeds, attention_mask).
"""


import pytest
import torch

from src.models.components import chronos_bolt_net
from src.models.components.chronos_bolt_net import ChronosBoltEncoderNet

B, T, M, D = 4, 12, 2, 384


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(D, D)

    def forward(self, h):
        return torch.tanh(self.lin(h))


class FakeBolt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_patch_embedding = torch.nn.Linear(32, D)
        self.encoder = torch.nn.Module()
        self.encoder.block = torch.nn.ModuleList(_Block() for _ in range(4))
        self.encoder.final_layer_norm = torch.nn.LayerNorm(D)
        self.reg = torch.nn.Parameter(torch.zeros(D))
        self.calls = 0

    def encode(self, context):
        self.calls += 1
        n, t = context.shape
        loc = torch.nan_to_num(torch.nanmean(context, dim=1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(torch.nanmean((context - loc) ** 2, dim=1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, torch.full_like(scale, 1e-5), scale)
        x = (context - loc) / scale
        pad = (16 - t % 16) % 16
        x = torch.cat([torch.full((n, pad), float("nan")), x], dim=1)
        patches = x.unfold(-1, 16, 16)
        mask = (~torch.isnan(patches)).float()
        h = self.input_patch_embedding(torch.cat([torch.nan_to_num(patches), mask], dim=-1))
        h = torch.cat([h, self.reg.expand(n, 1, D)], dim=1)
        for blk in self.encoder.block:
            h = blk(h)
        h = self.encoder.final_layer_norm(h)
        attn = torch.cat([mask.sum(-1) > 0, torch.ones(n, 1, dtype=torch.bool)], dim=1)
        return h, (loc, scale), h, attn


def _net(**kw):
    kw.setdefault("modalities", ["step", "distance"])
    net = ChronosBoltEncoderNet(**kw)
    fake = FakeBolt()
    fake.requires_grad_(False)
    if not net.freeze_encoder:
        for blk in list(fake.encoder.block)[-net.finetune_blocks:]:
            blk.requires_grad_(True)
        fake.encoder.final_layer_norm.requires_grad_(True)
    net.model = fake
    net.init_input_size(2 * M + 2, num_time_features=2)
    return net


def _x(second="subject"):
    torch.manual_seed(0)
    parts = [torch.rand(B, T, M) * 5]
    if second == "subject":
        parts.append(torch.randn(B, T, M))
    elif second == "mask":
        parts.append((torch.rand(B, T, M) > 0.3).float())
    parts.append(torch.randn(B, T, 2))
    return torch.cat(parts, dim=-1)


@pytest.fixture(autouse=True)
def _clear_cache():
    chronos_bolt_net._EMBED_CACHE.clear()


def test_sizes_concat_modalities_plus_stats_and_time():
    net = _net()
    assert net(_x()).shape == (B, 2)
    assert net.fc.in_features == D * M + 5 * 2 * M + 2  # embedding + stats (values + subject copy) + time


def test_mean_pooling_ignores_padding_patches_and_reg():
    net = _net()
    hidden = torch.zeros(1, 3, D)
    hidden[0, 0] = 100.0  # an all-padding patch
    hidden[0, 1] = 1.0  # the observed patch
    hidden[0, 2] = -50.0  # [REG]
    attn = torch.tensor([[False, True, True]])
    assert torch.allclose(net.pool(hidden, attn), torch.ones(1, D))
    assert torch.allclose(_net(token_pooling="reg").pool(hidden, attn), torch.full((1, D), -50.0))


def test_only_values_are_encoded_and_mask_gives_nan():
    net = _net()
    x = _x()
    assert torch.equal(net.encoder_input(x), x[:, :, :M].permute(0, 2, 1))
    mnet = _net(second_block="mask")
    xm = _x("mask")
    series = mnet.encoder_input(xm)
    mask = xm[:, :, M:2 * M].permute(0, 2, 1) > 0.5
    assert torch.isnan(series[~mask]).all() and not torch.isnan(series[mask]).any()
    assert not torch.isnan(mnet(xm)).any()


def test_cache_shared_across_fold_rescaling():
    net = _net()
    x = _x()
    net(x)
    calls = net.model.calls
    x2 = x.clone()
    x2[:, :, :M] = x2[:, :, :M] * 2.5 + 0.7
    net(x2)
    assert net.model.calls == calls


def test_frozen_head_trains_and_encoder_does_not():
    net = _net()
    torch.nn.functional.cross_entropy(net(_x()), torch.tensor([0, 1, 0, 1])).backward()
    assert net.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.model.parameters())


def test_finetuning_trains_only_the_last_blocks():
    net = _net(freeze_encoder=False, finetune_blocks=2, encoder_lr_scale=0.1)
    torch.nn.functional.cross_entropy(net(_x()), torch.tensor([0, 1, 0, 1])).backward()
    blocks = net.model.encoder.block
    assert all(b.lin.weight.grad is not None and b.lin.weight.grad.abs().sum() > 0 for b in blocks[-2:])
    assert all(b.lin.weight.grad is None for b in blocks[:-2])
    groups = net.optimizer_param_groups()
    assert groups[1]["lr_scale"] == 0.1 and not chronos_bolt_net._EMBED_CACHE


def test_repeated_setup_calls_keep_the_head():
    net = _net()
    w = net.fc.weight.clone()
    net.init_input_size(2 * M + 2, num_time_features=2)
    net.init_demographics(0)
    assert torch.equal(net.fc.weight, w)


def test_stats_only_ablation_never_loads_the_model():
    net = ChronosBoltEncoderNet(modalities=["step", "distance"], include_embedding=False)
    net.init_input_size(2 * M + 2, num_time_features=2)
    assert net(_x()).shape == (B, 2) and net.model is None


def test_settings_validated():
    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], token_pooling="last")
    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], freeze_encoder=False, finetune_blocks=0)
    with pytest.raises(ValueError):
        ChronosBoltEncoderNet(modalities=["step"], use_sequence_data=False)
    with pytest.raises(ValueError):
        _net(second_block="none")


def test_configs_instantiate():
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir

    from src.utils.resolvers import register_resolvers

    register_resolvers()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        frozen = compose(config_name="train", overrides=["model=chronos_bolt", "data/modalities=step_distance"])
        ft = compose(config_name="train", overrides=["model=chronos_bolt_ft", "data/modalities=step_distance"])
    a, b = hydra.utils.instantiate(frozen.model.net), hydra.utils.instantiate(ft.model.net)
    assert a.freeze_encoder and not b.freeze_encoder and b.finetune_blocks == 2 and b.encoder_lr_scale == 0.1


@pytest.mark.slow
def test_real_weights_match_the_packages_embed():
    """On the real checkpoint, our pooled embedding equals mean-pooling the observed patches of
    ChronosBoltPipeline.embed, and the fine-tuning path gives the same values."""
    pytest.importorskip("chronos")
    from chronos import ChronosBoltPipeline

    torch.manual_seed(0)
    series = torch.rand(6, M, 20) * 100
    frozen = ChronosBoltEncoderNet(modalities=["step", "distance"], cache_embeddings=False)
    frozen._ensure_model(torch.device("cpu"))
    ours = frozen._embed(series).reshape(6, M, -1)
    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-mini", device_map="cpu", torch_dtype=torch.float32)
    emb, _ = pipe.embed(series.reshape(-1, 20))  # [N, patches + 1, D]; 20 steps -> 2 patches, the first part-padded
    ref = emb[:, :-1, :].mean(dim=1).reshape(6, M, -1)  # both patches hold observed values
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()
    tuned = ChronosBoltEncoderNet(modalities=["step", "distance"], freeze_encoder=False, finetune_blocks=2)
    tuned._ensure_model(torch.device("cpu"))
    with torch.no_grad():
        assert torch.allclose(tuned._embed(series).reshape(6, M, -1), ref, atol=1e-4)
