"""Wraps Amazon's Chronos-Bolt time-series foundation models (e.g. `amazon/chronos-bolt-mini`)
as a `net` for `HealthLitModule` (``model=chronos_bolt``), set up like ``TiRexEncoderNet``.

Chronos-Bolt is a T5-style encoder-decoder forecasting model (mini: 4 encoder blocks, 384-d,
21M parameters). It standardises each series, cuts it into 16-step patches (left-padding with NaN
to a whole patch), embeds the patches, and appends a [REG] summary token. Unlike the OpenMHC
Chronos-2 checkpoint it needs no fixed weekly hourly grid and no channel vocabulary: any series
length and sampling rate is encoded directly.

Embedding extraction follows the same recipe as ``TiRexEncoderNet`` (Auer et al., 2025): every
modality is encoded separately, the final encoder states are mean-pooled over the patches that
hold observed values (``token_pooling="reg"`` takes the [REG] token instead), the per-modality
vectors are concatenated, and per-modality input statistics (``mean/std/min/max/last``, also of
the subject-scaled copy under ``DualScaler``) plus the survey-time features are appended so the
head sees the activity level that instance normalisation removes. Only the first
``len(modalities)`` columns are encoded; with ``second_block="mask"`` bins without records are
passed as NaN (Chronos-Bolt's missing-value marker).

The package's ``ChronosBoltPipeline.embed`` runs under ``torch.no_grad``, so this net calls
``model.encode`` directly: frozen embeddings are cached per series (keyed on standardised values,
see ``TiRexEncoderNet.series_keys``), and with ``freeze_encoder=False`` the last
``finetune_blocks`` encoder blocks train with the head at ``encoder_lr_scale`` x the optimizer lr.
Runs on CPU or GPU. Requires the ``chronos`` extra (``chronos-forecasting``).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.input_stats import INPUT_STAT_NAMES, input_stats
from src.models.components.tirex_net import TiRexEncoderNet

__all__ = ["DEFAULT_BOLT_MODEL_ID", "BOLT_EMBED_DIMS", "BOLT_NUM_BLOCKS", "ChronosBoltEncoderNet"]

DEFAULT_BOLT_MODEL_ID = "amazon/chronos-bolt-mini"

# d_model / encoder blocks from each checkpoint's config.json.
BOLT_EMBED_DIMS: Dict[str, int] = {
    "amazon/chronos-bolt-tiny": 256,
    "amazon/chronos-bolt-mini": 384,
    "amazon/chronos-bolt-small": 512,
    "amazon/chronos-bolt-base": 768,
}
BOLT_NUM_BLOCKS: Dict[str, int] = {
    "amazon/chronos-bolt-tiny": 4,
    "amazon/chronos-bolt-mini": 4,
    "amazon/chronos-bolt-small": 6,
    "amazon/chronos-bolt-base": 12,
}
TOKEN_POOLING_MODES = ("mean", "reg")
SECOND_BLOCK_MODES = ("subject", "mask", "none")

# Per-process cache of frozen per-series embeddings (see TiRexEncoderNet).
_EMBED_CACHE: Dict[Tuple, Dict[bytes, torch.Tensor]] = {}


class ChronosBoltEncoderNet(nn.Module):
    """Chronos-Bolt embedding (frozen or partly fine-tuned) + input statistics + linear head."""

    def __init__(
        self,
        modalities: List[str],
        model_id: str = DEFAULT_BOLT_MODEL_ID,
        output_size: int = 2,
        demographics_dim: int = 0,
        use_sequence_data: bool = True,
        freeze_encoder: bool = True,
        finetune_blocks: int = -1,
        encoder_lr_scale: float = 0.02,
        token_pooling: str = "mean",
        second_block: str = "subject",
        append_input_stats: bool = True,
        append_time_features: bool = True,
        include_embedding: bool = True,
        cache_embeddings: bool = True,
        batch_size: int = 1024,
        torch_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError("ChronosBoltEncoderNet requires sequence data (use_sequence_data=False is unsupported)")
        if token_pooling not in TOKEN_POOLING_MODES:
            raise ValueError(f"token_pooling must be one of {TOKEN_POOLING_MODES}, got {token_pooling!r}")
        if second_block not in SECOND_BLOCK_MODES:
            raise ValueError(f"second_block must be one of {SECOND_BLOCK_MODES}, got {second_block!r}")
        if not include_embedding and not append_input_stats:
            raise ValueError("include_embedding=False needs append_input_stats=True, or the head has no input")
        if not freeze_encoder and not include_embedding:
            raise ValueError("freeze_encoder=False needs include_embedding=True (there is nothing to fine-tune)")
        n_blocks = BOLT_NUM_BLOCKS.get(model_id, 4)
        if not -1 <= finetune_blocks <= n_blocks or finetune_blocks == 0:
            raise ValueError(f"finetune_blocks must be -1 (all) or 1..{n_blocks}, got {finetune_blocks}")

        self.modalities = list(modalities)
        self.model_id = model_id
        self.freeze_encoder = freeze_encoder
        self.n_blocks = n_blocks
        self.finetune_blocks = n_blocks if finetune_blocks == -1 else finetune_blocks
        self.encoder_lr_scale = encoder_lr_scale
        self.token_pooling = token_pooling
        self.second_block = second_block
        self.append_input_stats = append_input_stats
        self.append_time_features = append_time_features
        self.include_embedding = include_embedding
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.demographics_dim = demographics_dim
        self.second_cols: Optional[slice] = None
        self.time_cols: Optional[slice] = None

        n_mod = len(self.modalities)
        self.embed_dim = BOLT_EMBED_DIMS.get(model_id, 384) * n_mod if include_embedding else 0
        self.representation_dim = self.embed_dim + (len(INPUT_STAT_NAMES) * n_mod if append_input_stats else 0)
        self.model = None  # loaded lazily on the first forward (or by optimizer_param_groups)
        self.fc = nn.Linear(self.representation_dim + demographics_dim, output_size)

    # -- sizes reported by the module's setup() (idempotent: setup also runs for test) --------------
    def init_input_size(self, input_size: int, num_time_features: int = 0) -> None:
        n_mod = len(self.modalities)
        num_time_features = num_time_features or 0
        n_sensor_cols = input_size - num_time_features
        if n_sensor_cols not in (n_mod, 2 * n_mod):
            warnings.warn(
                f"cannot split {input_size} columns into {n_mod} modalities (x1 or x2) plus "
                f"{num_time_features} time features; using only the first {n_mod} columns"
            )
            n_sensor_cols, num_time_features = n_mod, 0
        if n_sensor_cols == 2 * n_mod and self.second_block == "none":
            raise ValueError("x has a second block of sensor columns but second_block='none'")
        if n_sensor_cols == n_mod and self.second_block != "none":
            raise ValueError(f"second_block={self.second_block!r} but x has no second block of sensor columns")
        self.second_cols = slice(n_mod, 2 * n_mod) if n_sensor_cols == 2 * n_mod else None
        self.time_cols = slice(n_sensor_cols, input_size) if num_time_features else None

        stat_series = n_mod * (2 if self.second_block == "subject" else 1) if self.append_input_stats else 0
        extra_time = num_time_features if self.append_time_features else 0
        representation_dim = self.embed_dim + len(INPUT_STAT_NAMES) * stat_series + extra_time
        if representation_dim != self.representation_dim:
            self.representation_dim = representation_dim
            self.fc = nn.Linear(representation_dim + self.demographics_dim, self.fc.out_features)

    def init_demographics(self, demographics_dim: int) -> None:
        if demographics_dim > 0 and demographics_dim != self.demographics_dim:
            self.demographics_dim = demographics_dim
            self.fc = nn.Linear(self.representation_dim + demographics_dim, self.fc.out_features)

    def train(self, mode: bool = True):
        """Keeps the pretrained encoder in eval mode (no dropout) even when fine-tuning."""
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    # -- encoder ------------------------------------------------------------------------------------
    def _ensure_model(self, device: torch.device) -> None:
        if self.model is not None:
            return
        try:
            from chronos import ChronosBoltPipeline
        except ImportError as e:
            raise ImportError("ChronosBoltEncoderNet needs `chronos-forecasting`: `uv sync --extra chronos`.") from e
        pipeline = ChronosBoltPipeline.from_pretrained(
            self.model_id, device_map="cuda" if device.type == "cuda" else "cpu",
            torch_dtype=getattr(torch, self.torch_dtype, torch.float32),
        )
        model = pipeline.model
        model.eval()
        model.requires_grad_(False)
        if not self.freeze_encoder:
            blocks = list(model.encoder.block)
            for block in blocks[-self.finetune_blocks :]:
                block.requires_grad_(True)
            model.encoder.final_layer_norm.requires_grad_(True)
            if self.finetune_blocks == len(blocks):
                model.input_patch_embedding.requires_grad_(True)
        self.model = model
        self.add_module("encoder", model)

    def optimizer_param_groups(self) -> List[dict]:
        """Head at the optimizer lr; trainable encoder blocks at ``encoder_lr_scale`` x it."""
        groups = [{"params": list(self.fc.parameters())}]
        if not self.freeze_encoder:
            self._ensure_model(self.fc.weight.device)
            groups.append({"params": [p for p in self.model.parameters() if p.requires_grad],
                           "lr_scale": self.encoder_lr_scale})
        return groups

    def encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, M, T]`` series for the encoder; NaN where the has-data mask is 0."""
        values = x[:, :, : len(self.modalities)]
        if self.second_block == "mask" and self.second_cols is not None:
            values = torch.where(x[:, :, self.second_cols] > 0.5, values, torch.full_like(values, float("nan")))
        return values.permute(0, 2, 1).contiguous()

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[N, patches + 1, D]`` (last position = [REG]) -> ``[N, D]``."""
        if self.token_pooling == "reg":
            return hidden[:, -1, :]
        patch_states, patch_mask = hidden[:, :-1, :], attention_mask[:, :-1].to(hidden.dtype)
        denom = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (patch_states * patch_mask.unsqueeze(-1)).sum(dim=1) / denom

    def _series_embed(self, flat: torch.Tensor) -> torch.Tensor:
        """Univariate series ``[N, T]`` -> ``[N, D]`` (gradients flow if the caller allows them)."""
        device = next(self.model.parameters()).device
        hidden, _loc_scale, _embeds, attention_mask = self.model.encode(context=flat.to(device=device, dtype=torch.float32))
        return self.pool(hidden.float(), attention_mask)

    @torch.no_grad()
    def _encode_series(self, flat: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._series_embed(c).cpu() for c in torch.split(flat.float(), self.batch_size)], dim=0)

    def _cache_key(self) -> Tuple:
        return (self.model_id, self.token_pooling)

    def _embed(self, series: torch.Tensor) -> torch.Tensor:
        b, m, t = series.shape
        flat = series.reshape(b * m, t)
        if not self.freeze_encoder:
            return self._series_embed(flat).reshape(b, -1).to(series.device)
        if not self.cache_embeddings:
            return self._encode_series(flat).reshape(b, -1).to(series.device)
        cache = _EMBED_CACHE.setdefault(self._cache_key(), {})
        keys = TiRexEncoderNet.series_keys(flat)  # Chronos-Bolt also standardises every series itself
        missing = list({k: i for i, k in enumerate(keys) if k not in cache}.values())
        if missing:
            fresh = self._encode_series(flat[missing])
            for j, i in enumerate(missing):
                cache[keys[i]] = fresh[j]
        return torch.stack([cache[k] for k in keys], dim=0).reshape(b, -1).to(series.device)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_sensor = x[:, :, : len(self.modalities)]
        features = []
        if self.include_embedding:
            self._ensure_model(x.device)
            first = next(self.model.parameters())
            if first.device != x.device:
                self.model.to(x.device)  # in place: optimizer references to its parameters stay valid
            features.append(self._embed(self.encoder_input(x)))
        if self.append_input_stats:
            features.append(input_stats(x_sensor))
            if self.second_block == "subject" and self.second_cols is not None:
                features.append(input_stats(x[:, :, self.second_cols]))
        if self.append_time_features and self.time_cols is not None:
            features.append(x[:, -1, self.time_cols])
        if demographics is not None:
            if demographics.ndim == 1:
                demographics = demographics.unsqueeze(-1)
            features.append(demographics.to(x.device, dtype=x.dtype))
        return self.fc(torch.cat(features, dim=-1))


if __name__ == "__main__":
    print(ChronosBoltEncoderNet(modalities=["step", "distance"]))
