import pytest
import torch
from src.models.components.simple_transformer import SimpleTransformer
from src.models.health_module import HealthLitModule


def test_simple_transformer_forward_shape():
    model = SimpleTransformer(input_size=2, hidden_size=32, nhead=2, num_layers=1, output_size=2)
    x = torch.randn(4, 24, 2)
    out = model(x)
    assert out.shape == (4, 2)


@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_simple_transformer_pooling_strategies(pooling):
    model = SimpleTransformer(input_size=3, hidden_size=16, nhead=2, pooling=pooling)
    x = torch.randn(2, 12, 3)
    out = model(x)
    assert out.shape == (2, 2)


def test_simple_transformer_dynamic_input_size():
    model = SimpleTransformer(input_size=2, hidden_size=32, nhead=2)
    x_new = torch.randn(4, 24, 5)  # 5 features instead of 2
    out = model(x_new)
    assert out.shape == (4, 2)
    assert model.input_size == 5


def test_simple_transformer_demographics():
    model = SimpleTransformer(input_size=2, hidden_size=32, nhead=2)
    model.init_demographics(demographics_dim=4)

    x = torch.randn(4, 24, 2)
    demographics = torch.randn(4, 4)

    out = model(x=x, demographics=demographics)
    assert out.shape == (4, 2)


def test_simple_transformer_in_health_lit_module():
    net = SimpleTransformer(input_size=2, hidden_size=16, nhead=2, output_size=2)
    optimizer_partial = torch.optim.Adam
    lit_module = HealthLitModule(net=net, optimizer=optimizer_partial)
    lit_module.num_classes = 2

    x = torch.randn(8, 24, 2)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    user_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    batch = (x, y, user_idx)
    loss, preds, targets, logits = lit_module.model_step(batch)

    assert loss.shape == ()
    assert preds.shape == (8,)
    assert targets.shape == (8,)
    assert logits.shape == (8, 2)
