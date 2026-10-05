from types import SimpleNamespace

import pytest
import torch

from src.models.health_module import BaselineHealthModule


def _module_with_training_data(targets, user_indices, batch_size=3):
    """A user_rate baseline set up against a fake datamodule's training batches."""
    batches = [
        {"targets": torch.tensor(targets[i : i + batch_size]), "user_indices": torch.tensor(user_indices[i : i + batch_size])}
        for i in range(0, len(targets), batch_size)
    ]
    module = BaselineHealthModule(strategy="user_rate", num_classes=2)
    module._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(train_dataloader=lambda: batches),
        is_global_zero=True,
        progress_bar_callback=None,  # read by LightningModule.print
    )
    module.setup("fit")
    return module


def test_user_rate_learns_each_participants_training_rate():
    # Participant 1: 1/4 positive; participant 2: 3/3 positive; overall 4/7.
    module = _module_with_training_data([0, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2])
    torch.testing.assert_close(module.user_rates[1:], torch.tensor([0.25, 1.0]))
    assert module.overall_rate == pytest.approx(4 / 7)


def test_user_rate_logits_softmax_to_the_rate_with_overall_fallback():
    module = _module_with_training_data([0, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2])
    # Index 0 = participant absent from training; 9 = out of range; both -> overall rate.
    users = torch.tensor([1, 2, 0, 9])
    probs = torch.softmax(module(torch.zeros(4, 1, 1), users), dim=1)[:, 1]
    torch.testing.assert_close(probs, torch.tensor([0.25, 1 - 1e-4, 4 / 7, 4 / 7]))


def test_user_rate_requires_user_indices():
    module = _module_with_training_data([0, 1], [1, 1])
    with pytest.raises(ValueError):
        module(torch.zeros(2, 1, 1))


def test_model_step_passes_user_indices():
    module = _module_with_training_data([0, 1, 1, 1], [1, 1, 2, 2])
    batch = {"features": torch.zeros(2, 1, 1), "targets": torch.tensor([1, 0]), "user_indices": torch.tensor([2, 1])}
    _, preds, _, logits = module.model_step(batch)
    assert preds.tolist() == [1, 0]  # rates 1.0 and 0.5 -> argmax tie at 0.5 resolves to class 0
    torch.testing.assert_close(torch.softmax(logits, 1)[:, 1], torch.tensor([1 - 1e-4, 0.5]))
