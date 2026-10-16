"""Absolute input-window statistics appended to frozen foundation-model embeddings.

TiRex and Chronos-Bolt instance-normalise each series, so their embeddings cannot tell a
sedentary window from an active one with the same shape; these statistics restore that.
"""

import torch

INPUT_STAT_NAMES = ("mean", "std", "min", "max", "last")


def input_stats(x_sensor: torch.Tensor) -> torch.Tensor:
    """Per-modality ``mean/std/min/max/last`` of the input window, ``[B, 5 * M]``."""
    stats = [
        x_sensor.mean(dim=1),
        x_sensor.std(dim=1, unbiased=False),
        x_sensor.amin(dim=1),
        x_sensor.amax(dim=1),
        x_sensor[:, -1, :],
    ]
    return torch.stack(stats, dim=-1).reshape(x_sensor.shape[0], -1)
