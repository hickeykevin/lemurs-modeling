import torch
import pytest
import numpy as np
from src.models.health_module import HealthLitModule, _resolve_num_classes
from src.models.components.simple_lstm import SimpleLSTM

def test_health_lit_module():
    """
    Tests the HealthLitModule logic using dummy data to ensure 
    the forward pass and training steps work correctly.
    """
    batch_size = 4
    time_steps = 24
    input_size = 2
    num_classes = 5
    
    # 1. Instantiate the underlying network
    net = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes
    )
    
    # 2. Instantiate the LightningModule
    # We pass a simple optimizer function for the test
    def optimizer_func(params):
        return torch.optim.Adam(params, lr=0.001)
        
    module = HealthLitModule(net=net, optimizer=optimizer_func)
    module.setup(stage="fit", num_classes=num_classes)
    
    # 3. Create a dummy batch
    # x: [Batch, Time, Features]
    x = torch.randn(batch_size, time_steps, input_size)
    # y: [Batch] (Target labels 0-4)
    y = torch.randint(0, num_classes, (batch_size,))
    # user_idx: [Batch]
    user_idx = torch.zeros(batch_size, dtype=torch.long)
    batch = (x, y, user_idx)
    
    # 4. Test forward pass
    output = module(x)
    assert output.shape == (batch_size, num_classes)
    
    # 5. Test training step
    loss = module.training_step(batch, 0)
    assert loss is not None
    assert loss.shape == torch.Size([]) # Must be a scalar tensor
    
    # 6. Test validation / loss metrics
    module.validation_step(batch, 0)
    val_loss = module.val_loss.compute()
    
    assert val_loss >= 0


def test_health_lit_module_ablation():
    """Tests SimpleLSTM sequence data ablation configuration."""
    batch_size = 4
    time_steps = 24
    input_size = 2
    num_classes = 5

    net_seq_only = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes,
        use_sequence_data=True
    )
    assert net_seq_only.fc.in_features == 16 # Just the hidden size
    
    x = torch.randn(batch_size, time_steps, input_size)
    output_seq = net_seq_only(x)
    assert output_seq.shape == (batch_size, num_classes)


def test_resolve_num_classes_prefers_explicit():
    """An explicit num_classes always wins, regardless of what else is available."""
    assert _resolve_num_classes(trainer=None, explicit=3, scan_fallback_labels=lambda: 99) == 3


def test_resolve_num_classes_prefers_aggregator_over_scan():
    """The aggregator's declared count must win even when the scan disagrees.

    This is the exact scenario that breaks under repeated grouped CV: a fold's
    validation split can be single-class purely by chance (the inner val carve-out
    is re-stratified from an already-shrunken user pool and can fall back to
    unstratified grouping), which would make a naive label scan return 1 instead
    of the true class count.
    """
    class FakeAggregator:
        num_classes = 2

    class FakeHparams:
        aggregator = FakeAggregator()

    class FakeDataModule:
        hparams = FakeHparams()

    class FakeTrainer:
        datamodule = FakeDataModule()

    # The scan simulates a single-class validation fold: it would return 1.
    result = _resolve_num_classes(FakeTrainer(), None, lambda: 1)
    assert result == 2


def test_resolve_num_classes_falls_back_to_scan_without_aggregator():
    """With no aggregator to consult, the scan is still the last resort."""
    class FakeDataModule:
        hparams = object()  # no .aggregator attribute

    class FakeTrainer:
        datamodule = FakeDataModule()

    assert _resolve_num_classes(FakeTrainer(), None, lambda: 5) == 5


def test_resolve_num_classes_no_trainer_uses_scan():
    assert _resolve_num_classes(None, None, lambda: 4) == 4


def test_health_lit_module_num_classes_survives_single_class_val_fold():
    """Regression test for the CV correctness bug: a fold whose validation split
    happens to contain only one class must not shrink num_classes below what the
    aggregator (and therefore the network's actual output layer) declare.
    """
    num_classes = 2
    net = SimpleLSTM(input_size=2, hidden_size=8, num_layers=1, output_size=num_classes)

    def optimizer_func(params):
        return torch.optim.Adam(params, lr=0.001)

    module = HealthLitModule(net=net, optimizer=optimizer_func)

    class FakeAggregator:
        num_classes = 2

    class FakeHparams:
        aggregator = FakeAggregator()

    class FakeDataModule:
        hparams = FakeHparams()

        def val_dataloader(self):
            # Every label in this fold's validation split is class 0 -- the
            # degenerate case a naive scan would misread as num_classes=1.
            x = torch.zeros(4, 24, 2)
            y = torch.zeros(4, dtype=torch.long)
            user_idx = torch.zeros(4, dtype=torch.long)
            return [(x, y, user_idx)]

    class FakeTrainer:
        datamodule = FakeDataModule()

    module.trainer = FakeTrainer()
    module.setup(stage="fit")

    assert module.num_classes == 2


if __name__ == "__main__":
    test_health_lit_module()

