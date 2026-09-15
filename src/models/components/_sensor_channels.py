"""Shared sensor-channel vocabulary + sampler-output windowing, used by every
pretrained-foundation-model `net` in this package (``wbm_net.py``, ``chronos2_net.py``,
...). See ``wbm_net``'s module docstring for the full "channel-mapping caveat" this
implies -- it applies identically to every net built on this module, since they all
share the same pretrained-checkpoint channel vocabulary (Apple HealthKit-only: no
Android/Google Fit channel exists in this vocabulary at all).

Vendored channel list source: OpenMHC (MIT) -- https://github.com/AshleyLab/OpenMHC,
commit a2d67834d3cda721eac1cf02e583b3804d37621d, ``src/openmhc/_constants.py``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Ordered list of the pretrained encoders' 19 sensor channels (column order fixed by
# the checkpoints -- both WBM and Chronos-2 were pretrained/fine-tuned on this same
# channel vocabulary and hourly resolution).
SENSOR_CHANNELS: List[str] = [
    "iphone_steps",
    "iphone_distance",
    "iphone_flights",
    "watch_steps",
    "watch_distance",
    "watch_hr",
    "watch_energy",
    "sleep_asleep",
    "sleep_inbed",
    "workout_walking",
    "workout_cycling",
    "workout_running",
    "workout_other",
    "workout_mixed_cardio",
    "workout_strength",
    "workout_elliptical",
    "workout_hiit",
    "workout_functional",
    "workout_yoga",
]
N_SENSOR_CHANNELS = len(SENSOR_CHANNELS)  # 19
N_CONTINUOUS_CHANNELS = 7  # channels 0-6 are the ones WBM z-scores; 7-18 are identity
WEEK_HOURS = 168

# Our own mapping from this repo's modality names (configs/data/preprocessors/*) onto
# the pretrained encoders' channel vocabulary -- see the "channel-mapping caveat" in
# wbm_net.py's module docstring. Shared across every net built on this module.
DEFAULT_MODALITY_CHANNEL_MAP: Dict[str, str] = {
    "step": "iphone_steps",
    "distance": "iphone_distance",
    "calorie": "watch_energy",
}


def resample_and_pad_to_week(
    x_sensor: torch.Tensor,
    modalities: List[str],
    modality_channel_map: Dict[str, str],
    resample_freq_hours: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maps this repo's ``[B, T, len(modalities)]`` sampler output onto the pretrained
    checkpoints' fixed weekly ``(168-hour, 19-channel)`` grid.

    Each modality's bins are repeated across the hours they span (``resample_freq_hours``
    must be a whole number of hours), then the resulting hourly series is left-padded
    with "unobserved" hours if the lookback covers under a week, or truncated to the
    most recent 168 hours if it covers more. This is the resample/pad/truncate step
    shared by every net in this package; each net's own module formats the result into
    whatever tensor layout its pretrained checkpoint expects (e.g. WBM's mask-channel
    concatenation + z-scoring, or Chronos-2's channel-first NaN-for-missing layout).

    Args:
        x_sensor: ``[B, T, len(modalities)]``, channel order matching ``modalities``
            (i.e. with any trailing non-sensor columns, such as this repo's cyclic
            time features, already sliced off by the caller).
        modalities: names of ``x_sensor``'s columns, in order (e.g. ``["step", "calorie"]``).
        modality_channel_map: ``modalities`` name -> one of ``SENSOR_CHANNELS``. A
            modality with no entry is left fully unobserved in the output.
        resample_freq_hours: hours spanned by one input bin (e.g. 6.0 for the default
            ``rolling`` sampler's ``"6h"`` bins).

    Returns:
        ``(values, observed)``, each ``[B, 168, 19]``: ``values`` holds the repeated/
        padded raw (unnormalized) sensor readings (0 where unobserved), ``observed`` is
        a boolean mask, ``True`` where a value was actually mapped in (not padding, not
        an unmapped modality).
    """
    if not float(resample_freq_hours).is_integer():
        raise ValueError(f"resample_freq_hours must be a whole number of hours, got {resample_freq_hours}")
    hours_per_bin = int(resample_freq_hours)

    batch_size, num_bins, num_modalities = x_sensor.shape
    if num_modalities != len(modalities):
        raise ValueError(
            f"x_sensor has {num_modalities} channels but {len(modalities)} modality names were given"
        )

    device = x_sensor.device
    observed_hours = num_bins * hours_per_bin
    values = torch.zeros(batch_size, observed_hours, N_SENSOR_CHANNELS, device=device, dtype=x_sensor.dtype)
    observed = torch.zeros(batch_size, observed_hours, N_SENSOR_CHANNELS, device=device, dtype=torch.bool)

    for col_idx, name in enumerate(modalities):
        mapped = modality_channel_map.get(name)
        if mapped is None:
            continue
        if mapped not in SENSOR_CHANNELS:
            raise ValueError(f"modality_channel_map[{name!r}] = {mapped!r} is not one of SENSOR_CHANNELS")
        ch_idx = SENSOR_CHANNELS.index(mapped)

        series = torch.repeat_interleave(x_sensor[:, :, col_idx], hours_per_bin, dim=1)  # [B, observed_hours]
        values[:, :, ch_idx] = series
        observed[:, :, ch_idx] = True

    if observed_hours < WEEK_HOURS:
        pad = WEEK_HOURS - observed_hours
        values = F.pad(values, (0, 0, pad, 0))  # left-pad the time dimension
        observed = F.pad(observed, (0, 0, pad, 0), value=False)
    elif observed_hours > WEEK_HOURS:
        values = values[:, -WEEK_HOURS:, :]
        observed = observed[:, -WEEK_HOURS:, :]

    return values, observed
