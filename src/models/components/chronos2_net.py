"""Wraps the pretrained/fine-tuned OpenMHC "Chronos-2" encoder as a `net` for
``HealthLitModule``, so it can be swapped in for ``SimpleLSTM``/``SimpleTransformer``/
``WBMEncoderNet`` via ``model=chronos2``.

The checkpoint is Amazon's `amazon/chronos-2` time-series foundation model, LoRA
fine-tuned (rank 8, alpha 16, merged into the base weights) by OpenMHC on the
MyHeartCounts cohort for 24h-ahead forecasting of the same 19 hourly sensor channels
WBM uses (see ``wbm_net.py``). Despite that forecasting fine-tune, OpenMHC's own
downstream-classification benchmark reuses this exact checkpoint as a feature
extractor: discard the forecast, keep the encoder's last hidden state per channel,
mean-pool across channels, and feed that into a probe. This module replicates that
same "encoder as embedding extractor" path as a `net`, exactly the role
``WBMEncoderNet`` plays for the WBM checkpoint.

Checkpoint: https://huggingface.co/MyHeartCounts/openmhc-chronos2-fc (CC-BY-4.0).
Base model: https://huggingface.co/amazon/chronos-2 -- "Beyond Sensor Data" is NOT
the paper behind this one; Chronos-2 is Amazon's own time-series foundation model
line (Chronos / Chronos-Bolt / Chronos-2), unrelated to WBM's Apple/USC paper. Wrapper
approach + the ``_predict_last_latent`` extraction below are adapted from OpenMHC
(MIT) -- https://github.com/AshleyLab/OpenMHC, commit
a2d67834d3cda721eac1cf02e583b3804d37621d,
``src/downstream_evaluation/models/{chronos2,tsfm}.py``. ``_predict_last_latent``
itself is, per OpenMHC's own header, doubly-derived from Amazon's
``chronos-forecasting`` (Apache-2.0) via a retired fork adding a last-latent
extraction method never merged upstream -- see the function's docstring for the
exact chain. Unlike that vendored *architecture* (WBM's tiny Mamba2 files), here we
depend on the real ``chronos-forecasting`` PyPI package for the model itself (it is
a full pretrained transformer, not something to vendor) and only port the ~60-line
extraction helper that isn't part of its public API.

IMPORTANT -- hardware: unlike WBM (CUDA-only ``mamba_ssm``), ``chronos-forecasting``
is a stock, pure-PyTorch package with no custom kernels. It runs on CPU (slow) or
GPU, so a real forward pass -- not just the adapter logic -- can be validated on a
laptop before ever touching the SLURM cluster. See the ``chronos2`` extra in
``pyproject.toml``.

IMPORTANT -- fine-tuning is not supported yet: extraction round-trips through
``Chronos2Dataset`` as a plain numpy array (matching OpenMHC's own extraction path
exactly), which breaks autograd. ``freeze_encoder=False`` therefore raises rather
than silently training nothing -- true end-to-end fine-tuning would need a different,
differentiable path through ``pipeline.model.encode`` and isn't implemented here.

IMPORTANT -- channel-mapping caveat: identical to ``WBMEncoderNet``'s -- see that
module's docstring. Both nets share the same 19-channel Apple HealthKit-only
vocabulary (``src/models/components/_sensor_channels.py``), so the same iOS-only
pretraining-distribution caveat applies to this net too.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.components._sensor_channels import (
    DEFAULT_MODALITY_CHANNEL_MAP,
    WEEK_HOURS,
    resample_and_pad_to_week,
)

__all__ = [
    "DEFAULT_CHECKPOINT_REPO_ID",
    "CHRONOS2_EMBED_DIM",
    "PREDICTION_LENGTH",
    "predict_last_latent",
    "Chronos2EncoderNet",
]

DEFAULT_CHECKPOINT_REPO_ID = "MyHeartCounts/openmhc-chronos2-fc"

# Confirmed directly from the checkpoint's own `checkpoint/config.json` (Chronos2CoreConfig
# .d_model) -- not from the model card, which doesn't state it. This is the base
# `amazon/chronos-2` architecture's hidden size; the LoRA fine-tune doesn't change it.
CHRONOS2_EMBED_DIM = 768

# OpenMHC's own default for this extraction (`downstream_evaluation.models.chronos2
# .PREDICTION_LENGTH`) -- a forecasting-pipeline artifact of calling `.encode(...)`,
# largely irrelevant to the embedding we actually keep (the last hidden state), but
# it does size `num_output_patches` internally, so we pass the same value upstream did.
PREDICTION_LENGTH = 24


def _resolve_checkpoint_dir(repo_id: str, revision: Optional[str] = None) -> Path:
    """Snapshot-downloads the OpenMHC release bundle and returns the actual Chronos-2
    checkpoint directory named by its manifest -- mirrors OpenMHC's own
    ``chronos2._download_hf_release`` rather than hardcoding the ``checkpoint/``
    subdirectory name, in case a future release renames it."""
    import json

    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["openmhc_manifest.json", "checkpoint/**", "config.json", "*.safetensors"],
        )
    )
    manifest = json.loads((local / "openmhc_manifest.json").read_text())
    return (local / manifest["checkpoint"]).resolve()


# --------------------------------------------------------------------------------- #
# Vendored/adapted extraction. Not part of Chronos-2's public API: OpenMHC re-derived
# it against the stock `chronos-forecasting==2.3.0` release because the method
# (`predict_last_latent`) only ever existed in a retired, never-merged fork. See
# OpenMHC's `chronos2.py` header for the full provenance chain. Trimmed here of the
# `cross_learning` knob (unused by this net's call site).
#
# Derived from amazon/chronos-forecasting (Apache-2.0):
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: Apache-2.0
# via OpenMHC (MIT) -- see module docstring for source commit.
# --------------------------------------------------------------------------------- #
def predict_last_latent(
    pipeline: Any,
    inputs,
    prediction_length: int,
    batch_size: int,
    context_length: int,
    limit_prediction_length: bool = False,
) -> List[torch.Tensor]:
    """Last hidden state of the final Chronos-2 encoder layer, per input item.

    Returns a list of tensors, each shape ``(n_target_variates, d_model)`` -- one per
    item in ``inputs``, using the same batching path ``predict``/``predict_quantiles``
    use internally. Caller wraps this in ``torch.no_grad()``; ``inputs`` should be a
    plain numpy array (this round-trips through ``Chronos2Dataset``, which does not
    preserve a torch autograd graph -- see the module docstring's fine-tuning caveat).
    """
    from torch.utils.data import DataLoader

    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    model_prediction_length = pipeline.model_prediction_length
    max_output_patches = pipeline.max_output_patches

    if prediction_length > model_prediction_length:
        msg = (
            f"We recommend keeping prediction length <= {model_prediction_length}. "
            "The quality of longer predictions may degrade since the model is not optimized for it."
        )
        if limit_prediction_length:
            raise ValueError(msg + " Set limit_prediction_length=False to turn off this check.")
        warnings.warn(msg)

    if context_length > pipeline.model_context_length:
        warnings.warn(
            f"context_length {context_length} exceeds the model's max context length "
            f"{pipeline.model_context_length}; resetting to the max."
        )
        context_length = pipeline.model_context_length

    test_dataset = Chronos2Dataset(
        inputs,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        mode=DatasetMode.TEST,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        pin_memory=pipeline.model.device.type == "cuda",
        shuffle=False,
        drop_last=False,
    )

    all_last_latents: List[torch.Tensor] = []
    for batch in test_loader:
        assert batch["future_target"] is None
        batch_context = batch["context"].to(device=pipeline.model.device, dtype=torch.float32)
        batch_group_ids = batch["group_ids"].to(device=pipeline.model.device)
        batch_future_covariates = batch["future_covariates"]
        batch_target_idx_ranges = batch["target_idx_ranges"]

        num_output_patches = math.ceil(prediction_length / pipeline.model_output_patch_size)
        num_output_patches = min(num_output_patches, max_output_patches)

        encode_kwargs: Dict[str, Any] = {}
        if batch_future_covariates is not None:
            batch_future_covariates = batch_future_covariates.to(
                device=pipeline.model.device, dtype=torch.float32
            )
            output_size = num_output_patches * pipeline.model_output_patch_size
            if output_size > batch_future_covariates.shape[1]:
                fc_bsz = len(batch_future_covariates)
                padding_size = output_size - batch_future_covariates.shape[1]
                padding_tensor = torch.full(
                    (fc_bsz, padding_size), fill_value=torch.nan, device=batch_future_covariates.device
                )
                batch_future_covariates = torch.cat([batch_future_covariates, padding_tensor], dim=1)
            else:
                batch_future_covariates = batch_future_covariates[..., :output_size]
            encode_kwargs["future_covariates"] = batch_future_covariates

        encoder_outputs, *_ = pipeline.model.encode(
            context=batch_context,
            group_ids=batch_group_ids,
            num_output_patches=num_output_patches,
            **encode_kwargs,
        )
        last_latents = encoder_outputs.last_hidden_state[:, -1, :].to(dtype=torch.float32, device="cpu")
        all_last_latents.extend(last_latents[start:end] for (start, end) in batch_target_idx_ranges)

    return all_last_latents


class Chronos2EncoderNet(nn.Module):
    """Pretrained-Chronos-2-encoder `net` for `HealthLitModule` -- drop-in alternative
    to `SimpleLSTM`/`SimpleTransformer`/`WBMEncoderNet`.

    The pipeline is loaded lazily (on first ``forward``, not at construction), same as
    ``WBMEncoderNet``, so this class can be instantiated anywhere -- including a plain
    config-composition dry run -- without network access until a real forward pass
    runs. Unlike WBM, that real forward pass can also run on CPU (slowly): see the
    module docstring.
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
        prediction_length: int = PREDICTION_LENGTH,
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError(
                "Chronos2EncoderNet always requires sequence data (use_sequence_data=False is unsupported)"
            )
        if not freeze_encoder:
            raise ValueError(
                "Chronos2EncoderNet does not support fine-tuning yet: extraction round-trips through "
                "Chronos2Dataset as a numpy array, which has no autograd graph to backprop through. "
                "Use freeze_encoder=True (a linear probe on the frozen embedding)."
            )

        self.modalities = list(modalities)
        self.resample_freq_hours = resample_freq_hours
        self.freeze_encoder = freeze_encoder
        self.checkpoint_repo_id = checkpoint_repo_id
        self.modality_channel_map = dict(modality_channel_map or DEFAULT_MODALITY_CHANNEL_MAP)
        self.demographics_dim = demographics_dim
        self.prediction_length = prediction_length
        self.embed_dim = CHRONOS2_EMBED_DIM

        self.pipeline = None  # loaded lazily; not an nn.Module itself (see _ensure_pipeline)
        self.fc = nn.Linear(self.embed_dim + demographics_dim, output_size)

    def init_input_size(self, input_size: int) -> None:
        """Sanity-checks the datamodule's feature count against our modality list --
        see ``WBMEncoderNet.init_input_size`` (same rationale: fixed encoder input
        width, so this only catches config drift, it never resizes anything)."""
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
        calls ``.train()`` -- see ``WBMEncoderNet.train`` (same rationale)."""
        super().train(mode)
        if self.freeze_encoder and self.pipeline is not None:
            self.pipeline.model.eval()
        return self

    def _ensure_pipeline(self, device: torch.device) -> None:
        if self.pipeline is not None:
            return
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "Chronos2EncoderNet needs the `chronos-forecasting` package. Install the "
                "`chronos2` extra: `uv sync --extra chronos2`."
            ) from e

        ckpt_dir = _resolve_checkpoint_dir(self.checkpoint_repo_id)
        pipeline = Chronos2Pipeline.from_pretrained(str(ckpt_dir), device_map=str(device))
        pipeline.model.eval()
        if self.freeze_encoder:
            pipeline.model.requires_grad_(False)
        self.pipeline = pipeline
        self.add_module("encoder", pipeline.model)  # track it as a submodule (device moves, state_dict, ...)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_pipeline(x.device)

        x_sensor = x[:, :, : len(self.modalities)]
        values, observed = resample_and_pad_to_week(x_sensor, self.modalities, self.modality_channel_map, self.resample_freq_hours)
        assert values.shape[1] == WEEK_HOURS

        nan_fill = torch.full_like(values, float("nan"))
        x_chw = torch.where(observed, values, nan_fill).permute(0, 2, 1)  # [B, 19, 168], NaN = unobserved

        with torch.no_grad():
            latents = predict_last_latent(
                self.pipeline,
                x_chw.detach().cpu().numpy(),
                prediction_length=self.prediction_length,
                batch_size=x_chw.shape[0],
                context_length=WEEK_HOURS,
            )
        # latents: list of (19, embed_dim) -> (B, 19, embed_dim) -> channel-mean-pool,
        # matching OpenMHC's own uniform pooling (TSFMEncoder._load_task).
        embeddings = torch.stack(latents, dim=0).to(x.device)
        pooled = embeddings.mean(dim=1)  # (B, embed_dim)

        if demographics is not None:
            pooled = torch.cat([pooled, demographics], dim=-1)

        return self.fc(pooled)


if __name__ == "__main__":
    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    print(net)
