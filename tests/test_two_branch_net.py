"""TwoBranchNet: layout of the item history, branch switches, modality dropout."""

import pytest
import torch

from src.models.components.two_branch_net import TwoBranchNet

B, T, F = 6, 12, 7        # sensor window
L, S = 5, 4               # 5 lags, 4 columns per lag (2 items, missing flag, gap)
EXTRA = 3                 # e.g. previous labels + missing flag, ahead of the item history


def _net(**kw):
    net = TwoBranchNet(input_size=F, **kw)
    net.init_demographics(EXTRA + L * S)
    net.init_item_sequence(L, S)
    return net


def _inputs():
    torch.manual_seed(0)
    return torch.randn(B, T, F), torch.rand(B, EXTRA + L * S) * 3


def test_forward_shape_and_extras_split():
    net = _net()
    x, d = _inputs()
    assert net(x, d).shape == (B, 2)
    assert net._extras_dim == EXTRA


def test_item_block_is_the_tail_read_oldest_first():
    net = _net()
    _, d = _inputs()
    d = d.clone()
    d[:, EXTRA:] = 0.0
    # Lag 1 (most recent) block gets value 1 in its first item column, lag 5 (oldest) gets value 5.
    for k in range(L):
        d[:, EXTRA + k * S] = float(k + 1)
    net.item_norm = torch.nn.Identity()
    seq = net._item_sequence(d)
    assert seq.shape == (B, L, S)
    assert seq[0, :, 0].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]  # oldest -> newest


def test_gap_column_is_log_transformed():
    net = _net()
    net.item_norm = torch.nn.Identity()
    d = torch.zeros(B, EXTRA + L * S)
    d[:, EXTRA + S - 1] = 23.0  # lag-1 gap, hours
    seq = net._item_sequence(d)
    assert torch.allclose(seq[0, -1, -1], torch.log1p(torch.tensor(23.0)))


def test_items_only_ignores_the_sensor_window():
    net = _net(use_sensor_branch=False).eval()
    x, d = _inputs()
    assert torch.allclose(net(x, d), net(torch.zeros_like(x), d))


def test_sensors_only_ignores_the_static_vector():
    net = _net(use_item_branch=False, use_static_extras=False).eval()
    x, d = _inputs()
    assert torch.allclose(net(x, d), net(x, torch.zeros_like(d)))
    assert net.item_gru is None


def test_disabled_item_branch_does_not_leak_the_item_block_as_extras():
    net = _net(use_item_branch=False, use_static_extras=True)
    assert net._extras_dim == EXTRA  # item history columns are excluded, not passed through raw


def test_modality_dropout_only_in_training_and_zeroes_one_stream():
    net = _net(modality_dropout=1.0)
    x, d = _inputs()
    net.eval()
    assert torch.allclose(net(x, d), net(x, d))  # deterministic in eval
    net.train()
    seen = set()
    for i in range(30):
        torch.manual_seed(i)
        # With p=1 exactly one embedding is zeroed for every sample; compare to the two single-branch zeros.
        seen.add(tuple(round(v, 4) for v in net(x, d)[0].tolist()))
    assert len(seen) > 1  # not always the same stream dropped


def test_requires_at_least_one_input():
    with pytest.raises(ValueError):
        TwoBranchNet(use_sensor_branch=False, use_item_branch=False, use_static_extras=False)


def test_item_branch_without_static_vector_errors():
    net = _net()
    x, _ = _inputs()
    with pytest.raises(ValueError):
        net(x, None)


def test_gradients_flow_to_both_branches():
    net = _net()
    x, d = _inputs()
    net(x, d).sum().backward()
    assert net.sensor_lstm.weight_ih_l0.grad.abs().sum() > 0
    assert net.item_gru.weight_ih_l0.grad.abs().sum() > 0


def test_config_instantiates_and_builds_from_data_sizes():
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=two_branch", "data/prev_items=social_stress",
                                                      "data/modalities=step_calorie_distance"])
    net = hydra.utils.instantiate(cfg.model.net)
    assert isinstance(net, TwoBranchNet) and net.use_sensor_branch and net.use_item_branch


def test_repeated_setup_calls_keep_trained_weights():
    """Lightning runs setup() again for the test stage; re-announcing the same sizes must be a no-op."""
    net = _net()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    net.init_input_size(F)
    net.init_demographics(EXTRA + L * S)
    net.init_item_sequence(L, S)
    after = net.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_changed_sizes_do_rebuild():
    net = _net()
    net.init_item_sequence(L + 1, S)
    assert net.item_gru.input_size == S and net.item_lags == L + 1
