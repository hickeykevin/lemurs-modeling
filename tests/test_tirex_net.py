"""TiRexEncoderNet: pooling, missing-value handling, caching, layout, and a real-weights check.

The fast tests use a stand-in for the TiRex model that mimics ``_embed_context``: it
left-pads to 2,048 steps (64 patches of 32), returns ``[N, tokens, blocks, D]`` for the
unpadded tokens, and runs under ``torch.inference_mode`` like the real one.
"""

import math

import pytest
import torch

from src.models.components import tirex_net
from src.models.components.tirex_net import TIREX_EMBED_DIM, TIREX_NUM_BLOCKS, TiRexEncoderNet

B, T, M = 4, 12, 2  # batch, steps (72 h at 6 h bins), modalities


class _Cfg:
    input_patch_size = 32


class FakeTiRex(torch.nn.Module):
    """``_embed_context`` stand-in: token j, block l holds (nanmean of the series + l) in every dim,
    except the last token, which also adds the number of observed values -- so tests can tell
    which tokens were pooled and whether NaNs reached the model."""

    config = _Cfg()

    def __init__(self):
        super().__init__()
        self.calls = 0

    @torch.inference_mode()
    def _embed_context(self, context):
        self.calls += 1
        n, t = context.shape
        n_tokens = math.ceil(t / 32)
        level = torch.nan_to_num(torch.nanmean(context, dim=1), nan=0.0)  # [n]
        observed = (~torch.isnan(context)).sum(dim=1).float()
        hidden = torch.zeros(n, n_tokens, TIREX_NUM_BLOCKS, TIREX_EMBED_DIM)
        for l in range(TIREX_NUM_BLOCKS):
            hidden[:, :, l, :] = (level + l + 1.0)[:, None, None]
        hidden[:, -1, :, :] += observed[:, None, None]
        return hidden


def _net(**kw):
    kw.setdefault("modalities", ["step", "distance"])
    net = TiRexEncoderNet(**kw)
    net.model = FakeTiRex()
    return net


def _x(second="subject", n_time=2):
    torch.manual_seed(0)
    parts = [torch.rand(B, T, M) * 5]
    if second == "subject":
        parts.append(torch.randn(B, T, M))
    elif second == "mask":
        parts.append((torch.rand(B, T, M) > 0.3).float())
    parts.append(torch.randn(B, T, n_time))
    return torch.cat(parts, dim=-1)


@pytest.fixture(autouse=True)
def _clear_cache():
    tirex_net._EMBED_CACHE.clear()


@pytest.mark.parametrize("pool,per_mod", [("all", TIREX_EMBED_DIM * TIREX_NUM_BLOCKS), ("mean", TIREX_EMBED_DIM),
                                          ("last", TIREX_EMBED_DIM)])
def test_embedding_and_head_sizes(pool, per_mod):
    net = _net(layer_pooling=pool)
    net.init_input_size(2 * M + 2, num_time_features=2)
    emb = net._encode(net.encoder_input(_x()))
    assert emb.shape == (B, per_mod * M)
    # head: embedding + 5 stats x (values + subject copy) + 2 time features
    assert net.fc.in_features == per_mod * M + 5 * 2 * M + 2
    assert net(_x()).shape == (B, 2)


def test_pools_only_the_tokens_covering_the_input():
    net = _net(layer_pooling="last")
    hidden = torch.zeros(1, 64, TIREX_NUM_BLOCKS, TIREX_EMBED_DIM)
    hidden[:, -1] = 1.0  # only the final token is non-zero
    out = net.pool(hidden, n_steps=12)  # 12 steps -> 1 patch -> only the final token
    assert torch.allclose(out.norm(dim=-1), torch.ones(1))
    hidden[:, -2, :, : TIREX_EMBED_DIM // 2] = 5.0  # a different direction from the final token
    out2 = net.pool(hidden, n_steps=40)  # 40 steps -> 2 patches -> mean of the last two tokens
    assert not torch.allclose(out, out2)


def test_blocks_are_l2_normalised_before_pooling():
    net = _net(layer_pooling="all")
    hidden = torch.rand(3, 1, TIREX_NUM_BLOCKS, TIREX_EMBED_DIM) * 10
    out = net.pool(hidden, n_steps=12).reshape(3, TIREX_NUM_BLOCKS, TIREX_EMBED_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3, TIREX_NUM_BLOCKS), atol=1e-5)


def test_mask_turns_empty_bins_into_nan_for_the_encoder():
    net = _net(second_block="mask")
    net.init_input_size(2 * M + 2, num_time_features=2)
    x = _x("mask")
    series = net.encoder_input(x)  # [B, M, T]
    mask = x[:, :, M : 2 * M].permute(0, 2, 1) > 0.5
    assert torch.isnan(series[~mask]).all() and not torch.isnan(series[mask]).any()
    # stats are computed on the raw values (zeros for empty bins), never NaN
    assert not torch.isnan(net(x)).any()


def test_subject_block_is_not_fed_to_the_encoder():
    net = _net(second_block="subject")
    net.init_input_size(2 * M + 2, num_time_features=2)
    x = _x("subject")
    assert torch.equal(net.encoder_input(x), x[:, :, :M].permute(0, 2, 1))


def test_cache_avoids_re_encoding():
    net = _net()
    net.init_input_size(2 * M + 2, num_time_features=2)
    x = _x()
    net(x)
    calls = net.model.calls
    net(x)
    assert net.model.calls == calls


def test_head_trains_on_inference_mode_embeddings():
    net = _net()
    net.init_input_size(2 * M + 2, num_time_features=2)
    loss = torch.nn.functional.cross_entropy(net(_x()), torch.tensor([0, 1, 0, 1]))
    loss.backward()  # would raise "Inference tensors cannot be saved for backward" without the clone
    assert net.fc.weight.grad is not None and net.fc.weight.grad.abs().sum() > 0


def test_repeated_setup_calls_keep_the_head():
    net = _net()
    net.init_input_size(2 * M + 2, num_time_features=2)
    w = net.fc.weight.clone()
    net.init_input_size(2 * M + 2, num_time_features=2)
    net.init_demographics(0)
    assert torch.equal(net.fc.weight, w)


def test_stats_only_ablation_never_loads_tirex():
    net = TiRexEncoderNet(modalities=["step", "distance"], include_embedding=False)
    net.init_input_size(2 * M + 2, num_time_features=2)
    assert net(_x()).shape == (B, 2) and net.model is None


def test_layout_mismatches_are_rejected():
    with pytest.raises(ValueError):
        _net(second_block="none").init_input_size(2 * M + 2, num_time_features=2)
    with pytest.raises(ValueError):
        _net(second_block="mask").init_input_size(M + 2, num_time_features=2)
    with pytest.raises(ValueError):
        TiRexEncoderNet(modalities=["step"], freeze_encoder=False)


def test_config_instantiates():
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir

    from src.utils.resolvers import register_resolvers

    register_resolvers()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=tirex", "data/modalities=step_distance"])
    net = hydra.utils.instantiate(cfg.model.net)
    assert isinstance(net, TiRexEncoderNet) and net.modalities == ["step", "distance"] and net.model is None


@pytest.mark.slow
def test_real_weights_match_the_packages_embedding():
    """With the real checkpoint, our per-modality embedding equals the TiRex package's own
    TiRexEmbedding (no augmentation) for the same series. Needs network/HF cache; run on a
    compute node: pytest -m slow tests/test_tirex_net.py"""
    pytest.importorskip("tirex")
    from tirex.models.embedding import TiRexEmbedding

    torch.manual_seed(0)
    series = torch.rand(6, M, T) * 100
    ref = TiRexEmbedding(device="cpu", data_augmentation=False)(series)
    net = TiRexEncoderNet(modalities=["step", "distance"], layer_pooling="all", cache_embeddings=False)
    net._ensure_model(torch.device("cpu"))
    ours = net._encode(series)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()


def test_cache_is_shared_across_fold_rescaling():
    """A fold refitting the scaler applies an affine map per modality; TiRex standardises each
    series itself, so the rescaled window must hit the cache, not be encoded again."""
    net = _net()
    net.init_input_size(2 * M + 2, num_time_features=2)
    x = _x()
    first = net(x)
    calls = net.model.calls
    x2 = x.clone()
    x2[:, :, :M] = x2[:, :, :M] * 3.7 - 1.2  # another fold's global scaler
    net(x2)
    assert net.model.calls == calls


def test_series_keys_scale_invariant_but_shape_sensitive():
    a = torch.tensor([[1.0, 2.0, float("nan"), 4.0]])
    keys = TiRexEncoderNet.series_keys
    assert keys(a) == keys(a * 10 + 3)
    assert keys(a) != keys(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))  # NaN pattern matters
    assert keys(a) != keys(torch.tensor([[4.0, 2.0, float("nan"), 1.0]]))


def test_cached_and_uncached_embeddings_agree():
    x = _x()
    a = _net(cache_embeddings=True)
    b = _net(cache_embeddings=False)
    for n in (a, b):
        n.init_input_size(2 * M + 2, num_time_features=2)
    assert torch.allclose(a._embed(a.encoder_input(x)), b._embed(b.encoder_input(x)))
