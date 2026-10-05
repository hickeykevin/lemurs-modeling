"""Wraps the pretrained/fine-tuned OpenMHC "Chronos-2" encoder as a `net` for
``HealthLitModule``, so it can be swapped in for ``SimpleLSTM``/``SimpleTransformer``/
``WBMEncoderNet`` via ``model=chronos2``.

The checkpoint is Amazon's `amazon/chronos-2` time-series foundation model, LoRA
fine-tuned (rank 8, alpha 16, merged into the base weights) by OpenMHC on the
MyHeartCounts cohort for 24h-ahead forecasting of the same 19 hourly sensor channels
WBM uses (see ``wbm_net.py``). Despite that forecasting fine-tune, OpenMHC's own
downstream-classification benchmark reuses this exact checkpoint as a feature
extractor: discard the forecast, keep the encoder's last hidden state per channel,
mean-pool across channels, and feed that into a probe. This module uses the same
"encoder as embedding extractor" path as a `net`, exactly the role ``WBMEncoderNet``
plays for the WBM checkpoint, but by default pools the embedding differently (see
``Chronos2EncoderNet``); OpenMHC's original pooling remains available via config.

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

IMPORTANT -- fine-tuning is not supported yet: embeddings are extracted under
``torch.no_grad()`` and cached per sample, so ``freeze_encoder=False`` raises rather
than silently training nothing. ``predict_last_latent`` (OpenMHC's numpy round-trip
through ``Chronos2Dataset``) is kept as the reference the direct ``encode`` path is
tested against.

IMPORTANT -- channel-mapping caveat: identical to ``WBMEncoderNet``'s -- see that
module's docstring. Both nets share the same 19-channel Apple HealthKit-only
vocabulary (``src/models/components/_sensor_channels.py``), so the same iOS-only
pretraining-distribution caveat applies to this net too.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.components._sensor_channels import (
    DEFAULT_MODALITY_CHANNEL_MAP,
    N_SENSOR_CHANNELS,
    SENSOR_CHANNELS,
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


# Per-process cache of frozen-encoder embeddings, keyed first by the extraction
# settings and then by the exact bytes of one sample's sensor input. The encoder is
# frozen and in eval mode, so a sample's embedding is a pure function of its input;
# without this, every training epoch re-runs the full Chronos-2 forward pass on data
# it has already embedded. Module-level so it survives the fresh `net` each CV fold
# builds (folds refit the scaler, so their inputs -- and keys -- differ anyway).
_EMBED_CACHE: Dict[Tuple, Dict[bytes, torch.Tensor]] = {}

TOKEN_POOLING_MODES = ("mean", "reg", "last")
LAYER_POOLING_MODES = ("last", "mean")
CHANNEL_POOLING_MODES = ("concat", "mean")
INPUT_STAT_NAMES = ("mean", "std", "min", "max", "last")


class Chronos2EncoderNet(nn.Module):
    """Pretrained-Chronos-2-encoder `net` for `HealthLitModule` -- drop-in alternative
    to `SimpleLSTM`/`SimpleTransformer`/`WBMEncoderNet`.

    The pipeline is loaded lazily (on first ``forward``, not at construction), same as
    ``WBMEncoderNet``, so this class can be instantiated anywhere -- including a plain
    config-composition dry run -- without network access until a real forward pass
    runs. Unlike WBM, that real forward pass can also run on CPU (slowly): see the
    module docstring.

    Embedding extraction follows Auer et al. (2025), "Pre-trained Forecasting Models:
    Strong Zero-Shot Feature Extractors for Time Series Classification", whose
    ablations found that, for frozen forecasting models used as classifiers:
    pooling over the input sequence beats taking a single position, concatenating
    per-variate embeddings beats averaging them, and appending absolute input
    statistics recovers the level information instance normalisation removes.

    The encoder's token sequence is ``[context patches | [REG] | output patches]``.
    The extraction knobs are:

    * ``token_pooling``: ``"mean"`` averages the final-layer states of the *observed*
      context patches (padding-only patches excluded); ``"reg"`` takes the [REG]
      summary token; ``"last"`` takes the final output-patch token (OpenMHC's choice).
    * ``layer_pooling``: ``"last"`` uses the final (normed) layer; ``"mean"`` averages
      every encoder block's output before token pooling.
    * ``channel_pooling``: ``"concat"`` keeps one embedding per channel; ``"mean"``
      averages them (OpenMHC's choice).
    * ``drop_unmapped_channels``: encode only the channels our modalities map to,
      rather than all 19 checkpoint channels with the unmapped ones fully missing.
      With ``False`` and ``channel_pooling="mean"``, 16 of the 19 averaged embeddings
      describe all-missing series.
    * ``append_input_stats``: concatenates per-modality ``mean/std/min/max/last`` of
      the (already scaled) input to the embedding.

    ``token_pooling="last", channel_pooling="mean", drop_unmapped_channels=False,
    append_input_stats=False`` reproduces OpenMHC's original extraction.
    """

    def __init__(
        self,
        modalities: List[str],
        output_size: int = 2,
        resample_freq_hours: Union[float, str] = 6.0,
        use_sequence_data: bool = True,
        demographics_dim: int = 0,
        freeze_encoder: bool = True,
        checkpoint_repo_id: str = DEFAULT_CHECKPOINT_REPO_ID,
        modality_channel_map: Optional[Dict[str, str]] = None,
        prediction_length: int = PREDICTION_LENGTH,
        token_pooling: str = "mean",
        layer_pooling: str = "last",
        channel_pooling: str = "concat",
        drop_unmapped_channels: bool = True,
        append_input_stats: bool = True,
        append_time_features: bool = True,
        cache_embeddings: bool = True,
        include_embedding: bool = True,
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError(
                "Chronos2EncoderNet always requires sequence data (use_sequence_data=False is unsupported)"
            )
        if not freeze_encoder:
            raise ValueError(
                "Chronos2EncoderNet does not support fine-tuning yet: the embedding is extracted "
                "under torch.no_grad() and cached. Use freeze_encoder=True (a linear probe on the "
                "frozen embedding)."
            )
        for name, value, allowed in (
            ("token_pooling", token_pooling, TOKEN_POOLING_MODES),
            ("layer_pooling", layer_pooling, LAYER_POOLING_MODES),
            ("channel_pooling", channel_pooling, CHANNEL_POOLING_MODES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

        self.modalities = list(modalities)
        # Accepts the sampler's own pandas frequency string (e.g. "6h") so the config can
        # interpolate `${data.sampler.resample_freq}` instead of restating it by hand.
        if isinstance(resample_freq_hours, str):
            import pandas as pd

            resample_freq_hours = pd.Timedelta(resample_freq_hours).total_seconds() / 3600.0
        self.resample_freq_hours = float(resample_freq_hours)
        self.freeze_encoder = freeze_encoder
        self.checkpoint_repo_id = checkpoint_repo_id
        self.modality_channel_map = dict(modality_channel_map or DEFAULT_MODALITY_CHANNEL_MAP)
        self.demographics_dim = demographics_dim
        self.prediction_length = prediction_length
        self.token_pooling = token_pooling
        self.layer_pooling = layer_pooling
        self.channel_pooling = channel_pooling
        self.drop_unmapped_channels = drop_unmapped_channels
        self.append_input_stats = append_input_stats
        self.append_time_features = append_time_features
        self.cache_embeddings = cache_embeddings
        # Located by init_input_size once the datamodule's column layout is known.
        self.subject_cols: Optional[slice] = None
        self.time_cols: Optional[slice] = None

        if drop_unmapped_channels:
            mapped = [self.modality_channel_map.get(m) for m in self.modalities]
            unknown = [c for c in mapped if c is not None and c not in SENSOR_CHANNELS]
            if unknown:
                raise ValueError(f"modality_channel_map maps to unknown channels {unknown}")
            # Modality order, de-duplicated (two modalities could share a channel).
            self.channel_idx = list(dict.fromkeys(SENSOR_CHANNELS.index(c) for c in mapped if c is not None))
            if not self.channel_idx:
                raise ValueError(f"none of {self.modalities} map to a checkpoint channel")
        else:
            self.channel_idx = list(range(N_SENSOR_CHANNELS))

        n_channels = len(self.channel_idx)
        # include_embedding=False is an ablation: the same head on input stats (and time
        # features) alone, to measure what the pretrained embedding adds beyond them.
        if not include_embedding and not append_input_stats:
            raise ValueError("include_embedding=False needs append_input_stats=True, or the head has no input")
        self.include_embedding = include_embedding
        self.embed_dim = (
            CHRONOS2_EMBED_DIM * (n_channels if channel_pooling == "concat" else 1) if include_embedding else 0
        )
        self.stats_dim = len(INPUT_STAT_NAMES) * len(self.modalities) if append_input_stats else 0
        self.representation_dim = self.embed_dim + self.stats_dim

        self.pipeline = None  # loaded lazily; not an nn.Module itself (see _ensure_pipeline)
        self.fc = nn.Linear(self.representation_dim + demographics_dim, output_size)

    def init_input_size(self, input_size: int, num_time_features: int = 0) -> None:
        """Locates the non-encoded input columns the head can also use.

        The encoder width is fixed, so nothing about it changes here. But the
        datamodule's column layout is ``[global-scaled sensors | subject-scaled
        sensors (DualScaler only) | time features]``, and the sequence baselines see
        all of it. Chronos-2 re-standardises every series, so encoding the
        subject-scaled copy would add nothing; its *statistics*, though (is this
        window high or low for this person?), and the time features at the survey
        are information the other models get, so the head gets them too.
        """
        n_mod = len(self.modalities)
        if input_size < n_mod:
            raise ValueError(
                f"datamodule produced {input_size} feature columns, fewer than the "
                f"{n_mod} modalities {self.modalities} this net expects"
            )
        n_sensor_cols = input_size - num_time_features
        if n_sensor_cols not in (n_mod, 2 * n_mod):
            warnings.warn(
                f"cannot split {input_size} columns into {n_mod} modalities (x1 or x2 for a "
                f"DualScaler) plus {num_time_features} time features; the head will only use "
                f"the first {n_mod} (global-scaled sensor) columns"
            )
            n_sensor_cols, num_time_features = n_mod, 0
        self.subject_cols = slice(n_mod, n_sensor_cols) if n_sensor_cols == 2 * n_mod else None
        self.time_cols = slice(n_sensor_cols, input_size) if num_time_features else None

        n_stat_series = n_sensor_cols if self.append_input_stats else 0
        extra_time = num_time_features if self.append_time_features else 0
        representation_dim = self.embed_dim + len(INPUT_STAT_NAMES) * n_stat_series + extra_time
        if representation_dim != self.representation_dim:
            self.representation_dim = representation_dim
            self.fc = nn.Linear(representation_dim + self.demographics_dim, self.fc.out_features)

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            self.fc = nn.Linear(self.representation_dim + demographics_dim, self.fc.out_features)

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

    def _cache_key(self) -> Tuple:
        return (
            self.checkpoint_repo_id,
            tuple(self.modalities),
            tuple(sorted(self.modality_channel_map.items())),
            float(self.resample_freq_hours),
            self.prediction_length,
            self.token_pooling,
            self.layer_pooling,
            self.channel_pooling,
            tuple(self.channel_idx),
        )

    @torch.no_grad()
    def _encode(self, x_sensor: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder embedding for ``x_sensor`` ``[B, T, len(modalities)]`` ->
        ``[B, embed_dim]`` (on CPU)."""
        model = self.pipeline.model
        device = model.device

        values, observed = resample_and_pad_to_week(
            x_sensor, self.modalities, self.modality_channel_map, self.resample_freq_hours
        )
        assert values.shape[1] == WEEK_HOURS
        context = torch.where(observed, values, torch.full_like(values, float("nan")))
        context = context[:, :, self.channel_idx].permute(0, 2, 1)  # [B, C, 168], NaN = unobserved
        batch_size, n_channels, length = context.shape

        # Channels of one sample share a group id, so Chronos-2's group attention mixes
        # information across them (its native multivariate mode) but never across samples.
        flat_context = context.reshape(batch_size * n_channels, length).to(device=device, dtype=torch.float32)
        group_ids = torch.arange(batch_size, device=device).repeat_interleave(n_channels)
        num_output_patches = min(
            math.ceil(self.prediction_length / self.pipeline.model_output_patch_size),
            self.pipeline.max_output_patches,
        )

        block_outputs: List[torch.Tensor] = []
        handles = []
        if self.layer_pooling == "mean":
            for block in model.encoder.block:
                handles.append(block.register_forward_hook(lambda _m, _i, out: block_outputs.append(out[0])))
        try:
            encoder_outputs, _loc_scale, _, num_context_patches = model.encode(
                context=flat_context, group_ids=group_ids, num_output_patches=num_output_patches
            )
        finally:
            for h in handles:
                h.remove()

        hidden = (
            torch.stack(block_outputs, dim=0).mean(dim=0)
            if self.layer_pooling == "mean"
            else encoder_outputs.last_hidden_state
        ).to(torch.float32)  # [B*C, num_context_patches + 1 + num_output_patches, d]

        if self.token_pooling == "last":
            tokens = hidden[:, -1, :]
        elif self.token_pooling == "reg":
            tokens = hidden[:, num_context_patches, :]
        else:
            patch_mask = self._observed_patch_mask(flat_context, num_context_patches).to(hidden.dtype)
            context_states = hidden[:, :num_context_patches, :]
            denom = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            tokens = (context_states * patch_mask.unsqueeze(-1)).sum(dim=1) / denom

        per_channel = tokens.reshape(batch_size, n_channels, -1)
        if self.channel_pooling == "concat":
            pooled = per_channel.reshape(batch_size, -1)
        else:
            pooled = per_channel.mean(dim=1)
        return pooled.cpu()

    def _observed_patch_mask(self, flat_context: torch.Tensor, num_patches: int) -> torch.Tensor:
        """``[N, num_patches]`` bool: True where a context patch holds at least one
        observed value -- the same rule Chronos-2 uses for its attention mask
        (``_prepare_patched_context``), which left-pads the context to whole patches."""
        patch = self.pipeline.model.chronos_config.input_patch_size
        observed = ~torch.isnan(flat_context)
        pad = num_patches * patch - observed.shape[1]
        if pad > 0:
            observed = torch.nn.functional.pad(observed, (pad, 0), value=False)
        elif pad < 0:
            observed = observed[:, -pad:]
        return observed.reshape(observed.shape[0], num_patches, patch).any(dim=-1)

    def _embed(self, x_sensor: torch.Tensor) -> torch.Tensor:
        """``_encode`` with per-sample caching (see ``_EMBED_CACHE``)."""
        if not self.cache_embeddings:
            return self._encode(x_sensor).to(x_sensor.device)

        cache = _EMBED_CACHE.setdefault(self._cache_key(), {})
        rows = x_sensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        keys = [rows[i].numpy().tobytes() for i in range(rows.shape[0])]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            fresh = self._encode(x_sensor[missing])
            for j, i in enumerate(missing):
                cache[keys[i]] = fresh[j]
        return torch.stack([cache[k] for k in keys], dim=0).to(x_sensor.device)

    @staticmethod
    def input_stats(x_sensor: torch.Tensor) -> torch.Tensor:
        """Per-modality ``mean/std/min/max/last`` of the input window, ``[B, 5 * M]``.
        Chronos-2 instance-normalises each series, so its embedding cannot tell a
        sedentary window from an active one with the same shape; these restore that."""
        stats = [
            x_sensor.mean(dim=1),
            x_sensor.std(dim=1, unbiased=False),
            x_sensor.amin(dim=1),
            x_sensor.amax(dim=1),
            x_sensor[:, -1, :],
        ]
        return torch.stack(stats, dim=-1).reshape(x_sensor.shape[0], -1)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_sensor = x[:, :, : len(self.modalities)]
        features = []
        if self.include_embedding:
            self._ensure_pipeline(x.device)
            features.append(self._embed(x_sensor))
        if self.append_input_stats:
            features.append(self.input_stats(x_sensor))
            if self.subject_cols is not None:
                features.append(self.input_stats(x[:, :, self.subject_cols]))
        if self.append_time_features and self.time_cols is not None:
            features.append(x[:, -1, self.time_cols])  # time features of the bin ending at the survey
        if demographics is not None:
            features.append(demographics)
        return self.fc(torch.cat(features, dim=-1))


if __name__ == "__main__":
    net = Chronos2EncoderNet(modalities=["step", "calorie"], output_size=2)
    print(net)
