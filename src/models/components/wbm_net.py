"""Wraps the pretrained OpenMHC "WBM" (Wearable Behavior Model) encoder as a `net`
for ``HealthLitModule``, so it can be swapped in for ``SimpleLSTM``/``SimpleTransformer``
via ``model=wbm`` the same way any other architecture is.

WBM is a bi-directional Mamba2 contrastive self-supervised encoder pretrained on a
week of hourly wearable data (168 hours x 19 sensor channels, doubled to 38 with a
per-channel missingness mask) from Stanford's My Heart Counts cohort. It is the
OpenMHC project's open reimplementation of the encoder introduced in:

    "Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables
    Improve Health Predictions" (Apple/USC, 2025) -- https://arxiv.org/abs/2507.00191

Checkpoint + normalization stats: https://huggingface.co/MyHeartCounts/openmhc-wbm-dp
(CC-BY-4.0). Encoder source: https://github.com/AshleyLab/OpenMHC (MIT), commit
a2d67834d3cda721eac1cf02e583b3804d37621d,
``src/downstream_evaluation/models/wbm/{week_encoders_mamba2,tokenizers}.py``.
The two small architecture classes below (``HourPatchEmbedding``, ``BiMamba2Block``,
``Mamba2WeekEncoder``) are vendored (lightly trimmed of training-only code) from that
commit rather than pulled in as a package dependency, since the rest of that repo is a
large benchmark harness (data loaders, cohort stores, W&B-coupled config) we don't need
just to run inference with the encoder.

IMPORTANT — hardware requirement: the Mamba2 kernels (``mamba_ssm``) are CUDA-only.
This module imports fine on any machine (the heavy imports are lazy), but actually
running the encoder (a real forward pass, so anything past construction) requires a
CUDA GPU with ``mamba-ssm`` installed -- see the ``wbm`` extra in ``pyproject.toml``.
It will raise a clear ``ImportError``/``RuntimeError`` rather than silently falling
back to CPU. Use ``trainer=gpu`` on a cluster node, not local Mac development.

IMPORTANT — channel-mapping caveat: this repo's rolling-window sampler produces
whatever modalities ``data.modalities`` names (by default ``["step", "calorie"]``,
see configs/data/default.yaml), at whatever bin width ``data.sampler`` uses (e.g.
6-hour bins over a 96-hour lookback for the default ``rolling`` sampler config), while
the pretrained encoder expects an exact (168-hour, 19-named-channel) weekly tensor.
``assemble_weekly_tensor`` below bridges that gap by: (1) mapping each of our named
modalities onto one of the pretrained encoder's 19 named sensor channels via
``modality_channel_map`` -- an explicit, overridable config choice, not a verified
ground truth (e.g. we map "step" -> "iphone_steps" rather than "watch_steps"; if your
cohort's steps mostly come from a wrist wearable, override the map); (2) repeating
each of our coarser bins across the hours it spans; (3) left-padding with "missing"
hours if our lookback is under a week, or keeping only the most recent 168 hours if
it's longer. Modalities with no entry in ``modality_channel_map`` are left fully
"missing" for the encoder (this is a real information loss, not a bug: the pretrained
channel vocabulary doesn't cover every signal we track, e.g. there is no channel for
our "calorie" data other than the wrist-derived "watch_energy" proxy). Also note our
sampler currently zero-fills empty bins rather than tracking true missingness, so the
"observed" mask this adapter produces means "we mapped a value here", not "the raw
sensor definitely had a reading here" -- a known approximation relative to how the
encoder was pretrained.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------- #
# Vendored from OpenMHC (MIT license) -- see module docstring for exact source.
# --------------------------------------------------------------------------------- #

# Ordered list of the pretrained encoder's 19 sensor channels (column order fixed by
# the checkpoint). Source: openmhc._constants.SENSOR_CHANNELS.
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
N_CONTINUOUS_CHANNELS = 7  # channels 0-6 are z-scored; 7-18 pass through as identity

# The published checkpoint's architecture (openmhc_manifest.json / downstream_evaluation
# .models.wbm.model._ARCH). Loading is strict=False with a missing-keys check, so a wrong
# dim here fails loudly rather than silently loading garbage.
WBM_ARCH = dict(in_dim=38, embed_dim=256, hidden_dim=64, num_layers=4, proj_dim=128, dropout=0.223)

DEFAULT_CHECKPOINT_REPO_ID = "MyHeartCounts/openmhc-wbm-dp"

# Our own mapping from this repo's modality names (configs/data/preprocessors/*) onto
# the pretrained encoder's channel vocabulary -- see the "channel-mapping caveat" above.
DEFAULT_MODALITY_CHANNEL_MAP: Dict[str, str] = {
    "step": "iphone_steps",
    "distance": "iphone_distance",
    "calorie": "watch_energy",
}


class HourPatchEmbedding(nn.Module):
    """TST-style patch embedding used by WBM: LayerNorm -> Linear -> GELU -> Linear.

    Vendored verbatim from OpenMHC's ``downstream_evaluation.models.wbm.tokenizers``.
    """

    def __init__(self, in_dim: int = 38, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return h


class BiMamba2Block(nn.Module):
    """Bidirectional Mamba2 block with FFN and residual connections.

    Vendored from OpenMHC's ``downstream_evaluation.models.wbm.week_encoders_mamba2``.
    Requires ``mamba_ssm`` (CUDA-only) -- raises ``ImportError`` at construction if
    unavailable, matching the upstream behavior, so the failure is immediate and clear
    rather than surfacing as a cryptic error deep in a forward pass.
    """

    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except Exception as e:
            raise ImportError(
                "BiMamba2Block requires the CUDA-only `mamba-ssm` package. Install the "
                "`wbm` extra (`uv sync --extra wbm`) on a CUDA GPU node -- this cannot "
                "run on a CPU-only machine."
            ) from e

        self.fwd = Mamba2(d_model=d_model)
        self.bwd = Mamba2(d_model=d_model)
        self.proj = nn.Linear(2 * d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y_f = self.fwd(x)
        y_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        y = self.proj(torch.cat([y_f, y_b], dim=-1))
        x = self.norm1(x + self.drop1(y))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class Mamba2WeekEncoder(nn.Module):
    """Tokenizer + Bi-Mamba2 backbone + projection head.

    Input:  (B, 168, in_dim) -- default (B, 168, 38): 19 z-scored sensor values + 19
        missingness-mask channels, one row per hour of a week.
    Output: (h, r) where ``h`` (B, proj_dim) is the normalized contrastive-projection
        vector (pretraining-only, unused downstream) and ``r`` (B, embed_dim) is the
        pooled representation this wrapper actually uses.

    Vendored from OpenMHC's ``downstream_evaluation.models.wbm.week_encoders_mamba2``.
    """

    def __init__(
        self,
        in_dim: int = 38,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        proj_dim: int = 128,
        dropout: float = 0.05,
        ffn_mult: int = 4,
        proj_head_type: str = "mlp",
    ):
        super().__init__()
        self.tokenizer = HourPatchEmbedding(in_dim=in_dim, embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.backbone = nn.Sequential(
            *[BiMamba2Block(embed_dim, ffn_mult=ffn_mult, dropout=dropout) for _ in range(num_layers)]
        )

        if proj_head_type == "linear":
            self.proj_head = nn.Sequential(nn.Linear(embed_dim, proj_dim))
        else:  # "mlp" -- 3-layer projector
            proj_hidden = 4 * embed_dim
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(proj_hidden, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(embed_dim, proj_dim),
            )

    def forward(self, x: torch.Tensor, keep_mask: Optional[torch.Tensor] = None):
        tok = self.tokenizer(x)
        seq = self.backbone(tok)

        if keep_mask is not None:
            mask_expanded = keep_mask.unsqueeze(-1)
            seq_masked = seq * mask_expanded
            r = seq_masked.sum(dim=1) / keep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            r = seq.mean(dim=1)

        h_raw = self.proj_head(r)
        h = F.normalize(h_raw, dim=-1)
        return h, r


# --------------------------------------------------------------------------------- #
# Our code: the sampler-output -> weekly-tensor adapter, and the HealthLitModule net.
# --------------------------------------------------------------------------------- #


def assemble_weekly_tensor(
    x_sensor: torch.Tensor,
    modalities: List[str],
    modality_channel_map: Dict[str, str],
    resample_freq_hours: float,
    norm_means: Optional[torch.Tensor] = None,
    norm_stds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Maps this repo's ``[B, T, len(modalities)]`` sampler output onto the pretrained
    encoder's fixed ``[B, 168, 38]`` weekly input.

    Each modality's bins are repeated across the hours they span (``resample_freq_hours``
    must evenly divide into whole hours), then the resulting hourly series is
    left-padded with "missing" hours if the lookback covers under a week, or truncated
    to the most recent 168 hours if it covers more. See the module docstring's
    "channel-mapping caveat" for what this adapter does and does not preserve.

    Args:
        x_sensor: ``[B, T, len(modalities)]``, channel order matching ``modalities``
            (i.e. with any trailing non-sensor columns, such as this repo's cyclic
            time features, already sliced off by the caller).
        modalities: names of ``x_sensor``'s columns, in order (e.g. ``["step", "calorie"]``).
        modality_channel_map: ``modalities`` name -> one of ``SENSOR_CHANNELS``. A
            modality with no entry is left fully "missing" in the output.
        resample_freq_hours: hours spanned by one input bin (e.g. 6.0 for the default
            ``rolling`` sampler's ``"6h"`` bins).
        norm_means, norm_stds: per-continuous-channel (length ``N_CONTINUOUS_CHANNELS``)
            z-score constants from the checkpoint's ``normalization_stats.json``. If
            ``None``, mapped continuous channels are left unnormalized (with a caller-
            side warning -- see ``WBMEncoderNet``).

    Returns:
        ``[B, 168, 38]`` tensor: channels 0-18 are the (masked, z-scored where
        applicable) sensor values, 19-37 are the corresponding missingness mask
        (1 = missing, 0 = mapped/observed) -- the same layout the checkpoint was
        pretrained on.
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
    missing = torch.ones(batch_size, observed_hours, N_SENSOR_CHANNELS, device=device, dtype=x_sensor.dtype)

    for col_idx, name in enumerate(modalities):
        mapped = modality_channel_map.get(name)
        if mapped is None:
            continue
        if mapped not in SENSOR_CHANNELS:
            raise ValueError(f"modality_channel_map[{name!r}] = {mapped!r} is not one of SENSOR_CHANNELS")
        ch_idx = SENSOR_CHANNELS.index(mapped)

        series = torch.repeat_interleave(x_sensor[:, :, col_idx], hours_per_bin, dim=1)  # [B, observed_hours]
        if ch_idx < N_CONTINUOUS_CHANNELS and norm_means is not None and norm_stds is not None:
            series = (series - norm_means[ch_idx]) / norm_stds[ch_idx]

        values[:, :, ch_idx] = series
        missing[:, :, ch_idx] = 0.0

    week_hours = 168
    if observed_hours < week_hours:
        pad = week_hours - observed_hours
        values = F.pad(values, (0, 0, pad, 0))  # left-pad the time dimension
        missing = F.pad(missing, (0, 0, pad, 0), value=1.0)
    elif observed_hours > week_hours:
        values = values[:, -week_hours:, :]
        missing = missing[:, -week_hours:, :]

    return torch.cat([values, missing], dim=-1)  # [B, 168, 38]


class WBMEncoderNet(nn.Module):
    """Pretrained-WBM-encoder `net` for `HealthLitModule` -- drop-in alternative to
    `SimpleLSTM`/`SimpleTransformer`.

    The encoder is loaded lazily (on first ``forward``, not at construction) so this
    class can be instantiated anywhere -- including plain config composition/dry runs
    on a machine without a GPU -- without needing `mamba_ssm` or network access until
    a real forward pass actually runs. See the module docstring for the hardware
    requirement and the channel-mapping caveat.
    """

    def __init__(
        self,
        modalities: List[str],
        output_size: int = 2,
        resample_freq_hours: float = 6.0,
        use_sequence_data: bool = True,
        demographics_dim: int = 0,
        freeze_encoder: bool = True,
        checkpoint_repo_id: str = DEFAULT_CHECKPOINT_REPO_ID,
        modality_channel_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError("WBMEncoderNet always requires sequence data (use_sequence_data=False is unsupported)")

        self.modalities = list(modalities)
        self.resample_freq_hours = resample_freq_hours
        self.freeze_encoder = freeze_encoder
        self.checkpoint_repo_id = checkpoint_repo_id
        self.modality_channel_map = dict(modality_channel_map or DEFAULT_MODALITY_CHANNEL_MAP)
        self.demographics_dim = demographics_dim
        self.embed_dim = WBM_ARCH["embed_dim"]

        self.encoder: Optional[Mamba2WeekEncoder] = None
        self.register_buffer("_norm_means", None, persistent=False)
        self.register_buffer("_norm_stds", None, persistent=False)

        self.fc = nn.Linear(self.embed_dim + demographics_dim, output_size)

    def init_input_size(self, input_size: int) -> None:
        """Sanity-checks the datamodule's feature count against our modality list.

        Unlike ``SimpleLSTM``/``SimpleTransformer``, the encoder's input width is fixed
        by the pretrained checkpoint (38), so this does not resize anything -- it just
        catches a config drift (e.g. ``data.modalities`` no longer matching what this
        net was built with) loudly instead of silently mis-slicing channels later.
        """
        if input_size < len(self.modalities):
            raise ValueError(
                f"datamodule produced {input_size} feature columns, fewer than the "
                f"{len(self.modalities)} modalities {self.modalities} this net expects"
            )

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            self.fc = nn.Linear(self.embed_dim + demographics_dim, self.fc.out_features)

    def train(self, mode: bool = True):
        """Keeps the pretrained encoder in eval mode even when the LightningModule
        calls ``.train()``, so BatchNorm/Dropout inside it never switch on when frozen."""
        super().train(mode)
        if self.freeze_encoder and self.encoder is not None:
            self.encoder.eval()
        return self

    def _ensure_encoder(self, device: torch.device) -> None:
        if self.encoder is not None:
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "WBMEncoderNet needs `huggingface_hub` to fetch the pretrained checkpoint. "
                "Install the `wbm` extra: `uv sync --extra wbm`."
            ) from e

        ckpt_path = hf_hub_download(repo_id=self.checkpoint_repo_id, filename="model.ckpt")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)

        enc_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                enc_state[k[len("model."):]] = v
            elif k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v
            else:
                enc_state[k] = v

        # Auto-detect proj_head type from the checkpoint (linear vs 3-layer mlp),
        # matching upstream's `_load_wbm_encoder`.
        proj_head_type = "mlp"
        w = enc_state.get("proj_head.0.weight")
        if w is not None and w.shape[0] == WBM_ARCH["proj_dim"]:
            proj_head_type = "linear"

        encoder = Mamba2WeekEncoder(proj_head_type=proj_head_type, **WBM_ARCH)
        result = encoder.load_state_dict(enc_state, strict=False)
        if result.missing_keys:
            raise RuntimeError(
                f"Missing keys loading WBM checkpoint (arch mismatch?): {result.missing_keys}"
            )
        encoder.eval().to(device)
        if self.freeze_encoder:
            encoder.requires_grad_(False)
        self.encoder = encoder  # nn.Module assignment auto-registers it as a submodule

        try:
            stats_path = hf_hub_download(repo_id=self.checkpoint_repo_id, filename="normalization_stats.json")
            import json

            stats = json.loads(open(stats_path).read())
            self._norm_means = torch.tensor(stats["means"][:N_CONTINUOUS_CHANNELS], device=device)
            self._norm_stds = torch.tensor(stats["stds"][:N_CONTINUOUS_CHANNELS], device=device)
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "Could not load normalization_stats.json for WBM; mapped continuous "
                "channels will be fed unnormalized, which will hurt this pretrained "
                "encoder's representations."
            )

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_encoder(x.device)

        x_sensor = x[:, :, : len(self.modalities)]
        weekly_x = assemble_weekly_tensor(
            x_sensor,
            self.modalities,
            self.modality_channel_map,
            self.resample_freq_hours,
            norm_means=self._norm_means,
            norm_stds=self._norm_stds,
        )

        with torch.set_grad_enabled(not self.freeze_encoder):
            _, r = self.encoder(weekly_x)

        if demographics is not None:
            r = torch.cat([r, demographics], dim=-1)

        return self.fc(r)


if __name__ == "__main__":
    net = WBMEncoderNet(modalities=["step", "calorie"], output_size=2)
    print(net)
