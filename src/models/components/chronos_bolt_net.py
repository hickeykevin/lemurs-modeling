"""Wraps Amazon's Chronos-Bolt time-series foundation models (e.g. `amazon/chronos-bolt-mini`)
as a `net` for `HealthLitModule`.

Unlike the OpenMHC Chronos-2 checkpoint (`chronos2_net.py`), Chronos-Bolt:
1. Does NOT require a rigid 168-hour (1 week) window or 1-hour binning: it accepts arbitrary
   sequence lengths (T) and arbitrary sampling rates without artificial padding or repetition.
2. Does NOT depend on a hardcoded 19-channel Apple HealthKit schema: it encodes modalities
   channel-independently, so any set of wearable sensors can be passed directly.
3. Uses a lightweight patch-based transformer architecture that is 10x-100x faster than
   autoregressive models and runs seamlessly on CPU or GPU.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "DEFAULT_BOLT_MODEL_ID",
    "BOLT_EMBED_DIMS",
    "ChronosBoltEncoderNet",
]

DEFAULT_BOLT_MODEL_ID = "amazon/chronos-bolt-mini"

BOLT_EMBED_DIMS: Dict[str, int] = {
    "amazon/chronos-bolt-tiny": 256,
    "amazon/chronos-bolt-mini": 384,
    "amazon/chronos-bolt-small": 512,
    "amazon/chronos-bolt-base": 768,
}


class ChronosBoltEncoderNet(nn.Module):
    """Pretrained Chronos-Bolt encoder `net` for `HealthLitModule`.

    Extracts representations from Amazon's Chronos-Bolt foundation model,
    pools across patches and modalities, optionally concatenates static demographics,
    and maps through a trainable linear head for downstream classification.

    The pipeline is loaded lazily (on first `forward`), allowing instantiation and
    dry-runs without immediate model downloading.
    """

    def __init__(
        self,
        modalities: List[str],
        model_id: str = DEFAULT_BOLT_MODEL_ID,
        output_size: int = 2,
        demographics_dim: int = 0,
        use_sequence_data: bool = True,
        freeze_encoder: bool = True,
        pooling: str = "mean",
        modality_pooling: str = "mean",
        torch_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError(
                "ChronosBoltEncoderNet requires sequence data (use_sequence_data=False is unsupported)"
            )
        if pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
        if modality_pooling not in ("mean", "concat"):
            raise ValueError(f"modality_pooling must be 'mean' or 'concat', got {modality_pooling!r}")

        self.modalities = list(modalities)
        self.model_id = model_id
        self.output_size = output_size
        self.demographics_dim = demographics_dim
        self.freeze_encoder = freeze_encoder
        self.pooling = pooling
        self.modality_pooling = modality_pooling
        self.torch_dtype = torch_dtype

        # Base hidden size from known models or fallback to 384 (mini)
        self.embed_dim = BOLT_EMBED_DIMS.get(model_id, 384)
        self.representation_dim = (
            self.embed_dim * len(self.modalities) if modality_pooling == "concat" else self.embed_dim
        )

        self.pipeline = None  # Loaded lazily on first forward
        self.fc = nn.Linear(self.representation_dim + demographics_dim, output_size)

    def init_input_size(self, input_size: int) -> None:
        """Sanity-checks that input features contain at least our expected modalities."""
        if input_size < len(self.modalities):
            raise ValueError(
                f"datamodule produced {input_size} feature columns, fewer than the "
                f"{len(self.modalities)} modalities {self.modalities} this net expects"
            )

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            self.fc = nn.Linear(self.representation_dim + demographics_dim, self.fc.out_features)

    def train(self, mode: bool = True):
        """Keeps the pretrained encoder in eval mode when freeze_encoder=True."""
        super().train(mode)
        if self.freeze_encoder and self.pipeline is not None:
            self.pipeline.model.eval()
        return self

    def _ensure_pipeline(self, device: torch.device) -> None:
        if self.pipeline is not None:
            return
        try:
            from chronos import ChronosBoltPipeline
        except ImportError as e:
            raise ImportError(
                "ChronosBoltEncoderNet needs the `chronos-forecasting` package. "
                "Install the extra: `uv sync --extra chronos2`."
            ) from e

        dtype = getattr(torch, self.torch_dtype, torch.float32)
        device_str = "cuda" if device.type == "cuda" else "cpu"
        pipeline = ChronosBoltPipeline.from_pretrained(
            self.model_id,
            device_map=device_str,
            torch_dtype=dtype,
        )
        pipeline.model.eval()
        if self.freeze_encoder:
            pipeline.model.requires_grad_(False)

        # Update embed_dim dynamically if config differs
        actual_d_model = getattr(pipeline.model.config, "d_model", self.embed_dim)
        if actual_d_model != self.embed_dim:
            self.embed_dim = actual_d_model
            self.representation_dim = (
                self.embed_dim * len(self.modalities) if self.modality_pooling == "concat" else self.embed_dim
            )
            self.fc = nn.Linear(self.representation_dim + self.demographics_dim, self.output_size).to(device)

        self.pipeline = pipeline
        self.add_module("encoder", pipeline.model)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_pipeline(x.device)

        # Keep only sensor modalities (ignore any trailing cyclic time features)
        x_sensor = x[:, :, : len(self.modalities)]
        b, t, f = x_sensor.shape

        # Reshape to channel-independent batch of 1D series: [B * F, T]
        # Transpose so each modality is a separate univariate series
        x_flat = x_sensor.permute(0, 2, 1).reshape(b * f, t)

        # Extract embeddings using ChronosBoltPipeline's native embed() API
        if self.freeze_encoder:
            with torch.no_grad():
                embeds, _ = self.pipeline.embed(x_flat)
        else:
            embeds, _ = self.pipeline.embed(x_flat)

        # embeds shape: [B * F, num_patches, d_model]
        embeds = embeds.to(x.device, dtype=torch.float32)

        # 1. Pool across time patches
        if self.pooling == "last":
            patch_pooled = embeds[:, -1, :]  # [B * F, d_model]
        else:
            patch_pooled = embeds.mean(dim=1)  # [B * F, d_model]

        # 2. Reshape to [B, F, d_model] and pool across modalities
        mod_embeds = patch_pooled.view(b, f, -1)
        if self.modality_pooling == "concat":
            pooled = mod_embeds.reshape(b, -1)  # [B, F * d_model]
        else:
            pooled = mod_embeds.mean(dim=1)  # [B, d_model]

        # 3. Concatenate static demographics if available
        if demographics is not None:
            if demographics.ndim == 1:
                demographics = demographics.unsqueeze(-1)
            pooled = torch.cat([pooled, demographics.to(pooled.device, dtype=pooled.dtype)], dim=-1)

        return self.fc(pooled)
