"""Wraps NX-AI's pretrained TiRex forecasting model as a frozen feature extractor `net` for
``HealthLitModule`` (``model=tirex``), alongside ``Chronos2EncoderNet``.

TiRex (Auer et al., NeurIPS 2025, "TiRex: Zero-Shot Forecasting Across Long and Short
Horizons with Enhanced In-Context Learning") is a 35M-parameter xLSTM (12 sLSTM blocks,
512-d hidden state) pretrained for zero-shot forecasting. It reads a univariate series in
patches of 32 steps, left-padding the context with NaN to its 2,048-step training length;
NaN is its native missing-value marker. Checkpoint: https://huggingface.co/NX-AI/TiRex
(NXAI Community License: free for research; attribution notice when redistributing).

Embedding extraction follows the TiRex package's own ``TiRexEmbedding`` (the recipe of
Auer et al., 2025, "Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors
for Time Series Classification"): every modality is encoded separately, each block's
hidden states are mean-pooled over the patches that cover the input, each block's vector
is L2-normalised, the per-block / per-modality vectors are concatenated and layer-normed.
Two deliberate departures:

* The package's optional sample-statistics features are standardised with the statistics
  of whatever batch is passed in (``TiRexEmbedding._normalize_stats``), which here would
  mix test-set statistics into training features. Instead the same per-modality input
  statistics as ``Chronos2EncoderNet`` (``mean/std/min/max/last``) are appended raw, and
  the trained head learns their scale.
* ``layer_pooling`` can also average the blocks or keep only the last one, to keep the
  representation small (all 12 blocks x 512 = 6,144 dims per modality).

Input length: our windows are 2-30 steps at 4-12 h bins, so the whole window sits inside
one 32-step patch and the embedding comes from a single token per block. Hourly bins
(``resample_freq: 1h``) give 24-120 steps, i.e. 1-4 patches.

Column layout of ``x`` (from the datamodule): ``[sensor values (M) | second block (M) |
time features]``. The second block is the subject-scaled copy under ``DualScaler``
(``second_block="subject"``) or the has-data mask under ``MissingMaskSampler``
(``second_block="mask"``); with ``"mask"``, bins with no records are passed to TiRex as
NaN rather than as zero activity.

The encoder is frozen and its embeddings cached per sample; fine-tuning is not supported.
Runs on CPU (the package's pure-torch sLSTM) or GPU; the optional custom CUDA kernels
(``tirex-ts[cuda]``) are not needed.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.chronos2_net import INPUT_STAT_NAMES, Chronos2EncoderNet

__all__ = ["DEFAULT_TIREX_REPO_ID", "TIREX_EMBED_DIM", "TIREX_NUM_BLOCKS", "TiRexEncoderNet"]

DEFAULT_TIREX_REPO_ID = "NX-AI/TiRex"
# From the checkpoint's own hyper_parameters (block_kwargs.embedding_dim / num_blocks).
TIREX_EMBED_DIM = 512
TIREX_NUM_BLOCKS = 12

LAYER_POOLING_MODES = ("all", "mean", "last")
SECOND_BLOCK_MODES = ("subject", "mask", "none")

# Per-process cache of frozen per-series embeddings, keyed by extraction settings, then by
# the standardised values of one modality's window (see TiRexEncoderNet.series_keys).
_EMBED_CACHE: Dict[Tuple, Dict[bytes, torch.Tensor]] = {}


class TiRexEncoderNet(nn.Module):
    """Frozen TiRex embedding + input statistics + linear head.

    Args:
        modalities: modality names; the first ``len(modalities)`` columns of ``x``.
        layer_pooling: ``"all"`` concatenates every block (the TiRex package's recipe),
            ``"mean"`` averages the L2-normalised blocks, ``"last"`` keeps the final block.
        second_block: what columns ``M:2M`` hold, see the module docstring.
        append_input_stats: append per-modality ``mean/std/min/max/last`` of the input
            (and of the subject-scaled copy, if present).
        append_time_features: append the time features of the bin ending at the survey.
        include_embedding: ``False`` is the stats-only ablation (same head, no TiRex).
        batch_size: encoder batch size (series per forward pass).
    """

    def __init__(
        self,
        modalities: List[str],
        output_size: int = 2,
        use_sequence_data: bool = True,
        demographics_dim: int = 0,
        freeze_encoder: bool = True,
        checkpoint_repo_id: str = DEFAULT_TIREX_REPO_ID,
        layer_pooling: str = "all",
        second_block: str = "subject",
        append_input_stats: bool = True,
        append_time_features: bool = True,
        include_embedding: bool = True,
        cache_embeddings: bool = True,
        batch_size: int = 512,
    ) -> None:
        super().__init__()
        if not use_sequence_data:
            raise ValueError("TiRexEncoderNet always requires sequence data (use_sequence_data=False is unsupported)")
        if not freeze_encoder:
            raise ValueError("TiRexEncoderNet does not support fine-tuning: embeddings are extracted frozen and cached.")
        if layer_pooling not in LAYER_POOLING_MODES:
            raise ValueError(f"layer_pooling must be one of {LAYER_POOLING_MODES}, got {layer_pooling!r}")
        if second_block not in SECOND_BLOCK_MODES:
            raise ValueError(f"second_block must be one of {SECOND_BLOCK_MODES}, got {second_block!r}")
        if not include_embedding and not append_input_stats:
            raise ValueError("include_embedding=False needs append_input_stats=True, or the head has no input")

        self.modalities = list(modalities)
        self.freeze_encoder = freeze_encoder
        self.checkpoint_repo_id = checkpoint_repo_id
        self.layer_pooling = layer_pooling
        self.second_block = second_block
        self.append_input_stats = append_input_stats
        self.append_time_features = append_time_features
        self.include_embedding = include_embedding
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.demographics_dim = demographics_dim
        # Located by init_input_size once the datamodule's column layout is known.
        self.second_cols: Optional[slice] = None
        self.time_cols: Optional[slice] = None

        n_mod = len(self.modalities)
        per_mod = TIREX_EMBED_DIM * (TIREX_NUM_BLOCKS if layer_pooling == "all" else 1)
        self.embed_dim = per_mod * n_mod if include_embedding else 0
        self.representation_dim = self.embed_dim + (len(INPUT_STAT_NAMES) * n_mod if append_input_stats else 0)
        self.model = None  # loaded lazily on the first forward (see _ensure_model)
        self.fc = nn.Linear(self.representation_dim + demographics_dim, output_size)

    # -- sizes reported by the module's setup() (idempotent: setup also runs for test) --------------
    def init_input_size(self, input_size: int, num_time_features: int = 0) -> None:
        n_mod = len(self.modalities)
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

        stat_series = 0
        if self.append_input_stats:
            stat_series = n_mod * (2 if self.second_block == "subject" else 1)
        extra_time = num_time_features if self.append_time_features else 0
        representation_dim = self.embed_dim + len(INPUT_STAT_NAMES) * stat_series + extra_time
        if representation_dim != self.representation_dim:
            self.representation_dim = representation_dim
            self.fc = nn.Linear(representation_dim + self.demographics_dim, self.fc.out_features)

    def init_demographics(self, demographics_dim: int) -> None:
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            self.fc = nn.Linear(self.representation_dim + demographics_dim, self.fc.out_features)

    def train(self, mode: bool = True):
        """Keeps the pretrained encoder in eval mode even when Lightning calls ``.train()``."""
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    # -- encoder --------------------------------------------------------------------------------------
    def _ensure_model(self, device: torch.device) -> None:
        if self.model is not None:
            return
        try:
            from tirex import load_model
        except ImportError as e:
            raise ImportError("TiRexEncoderNet needs the `tirex-ts` package: `uv sync --extra tirex`.") from e
        model = load_model(self.checkpoint_repo_id, device=str(device), backend="torch")
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.add_module("encoder", model)

    def _cache_key(self) -> Tuple:
        return (self.checkpoint_repo_id, self.layer_pooling)

    def encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, M, T]`` series for the encoder; NaN where the has-data mask is 0."""
        values = x[:, :, : len(self.modalities)]
        if self.second_block == "mask" and self.second_cols is not None:
            observed = x[:, :, self.second_cols] > 0.5
            values = torch.where(observed, values, torch.full_like(values, float("nan")))
        return values.permute(0, 2, 1).contiguous()

    def pool(self, hidden: torch.Tensor, n_steps: int) -> torch.Tensor:
        """``hidden`` ``[N, tokens, blocks, D]`` -> ``[N, per_mod]``.

        Mean over the patches that cover the input (the package's rule:
        ``ceil(n_steps / patch)`` final tokens), L2-normalise each block, then pool blocks.
        """
        patch = self.model.config.input_patch_size if self.model is not None else 32
        n_patches = -(-n_steps // patch)
        tokens = hidden[:, -n_patches:, :, :].mean(dim=1)  # [N, blocks, D]
        tokens = F.normalize(tokens, p=2, dim=-1)
        if self.layer_pooling == "last":
            return tokens[:, -1, :]
        if self.layer_pooling == "mean":
            return tokens.mean(dim=1)
        return tokens.flatten(start_dim=1)

    @torch.no_grad()
    def _encode_series(self, flat: torch.Tensor) -> torch.Tensor:
        """Univariate series ``[N, T]`` -> pooled per-series embeddings ``[N, per_mod]`` (CPU)."""
        out = []
        for chunk in torch.split(flat.float(), self.batch_size):
            hidden = self.model._embed_context(chunk)  # [n, tokens, blocks, D]
            out.append(self.pool(hidden.float(), flat.shape[1]).cpu())
        # TiRex's _embed_context runs under torch.inference_mode; clone so the head's
        # backward pass can save this input (inference tensors cannot be saved for backward).
        return torch.cat(out, dim=0).clone()

    def _assemble(self, per_series: torch.Tensor, b: int, m: int) -> torch.Tensor:
        """``[B*M, per_mod]`` -> ``[B, embed_dim]``: concatenate modalities (block-major for
        ``layer_pooling="all"``, as ``TiRexEmbedding.process_embedding`` flattens) and layer-norm."""
        per_mod = per_series.reshape(b, m, -1)
        if self.layer_pooling == "all":
            per_mod = per_mod.reshape(b, m, TIREX_NUM_BLOCKS, TIREX_EMBED_DIM).transpose(1, 2)
        emb = per_mod.reshape(b, -1)
        return F.layer_norm(emb, (emb.shape[-1],))

    def _encode(self, series: torch.Tensor) -> torch.Tensor:
        """``series`` ``[B, M, T]`` -> ``[B, embed_dim]`` (on CPU), no caching."""
        b, m, t = series.shape
        return self._assemble(self._encode_series(series.reshape(b * m, t)), b, m)

    @staticmethod
    def series_keys(flat: torch.Tensor) -> List[bytes]:
        """One cache key per univariate series, invariant to affine rescaling.

        TiRex standardises every series itself (``PatchedTokenizer``: NaN-aware mean and
        RMS deviation, with ``|mean| + 1e-5`` when the deviation is 0), so a window's
        embedding does not change when a CV fold refits our scaler. Keying on the
        standardised values (rounded, NaN kept) lets every fold, and every modality
        combination, reuse one encoding of the same window.
        """
        flat = flat.detach().to(device="cpu", dtype=torch.float64)
        loc = torch.nan_to_num(torch.nanmean(flat, dim=1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(torch.nanmean((flat - loc).square(), dim=1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, loc.abs() + 1e-5, scale)
        z = torch.round((flat - loc) / scale, decimals=4).to(torch.float32).numpy()
        return [z[i].tobytes() for i in range(z.shape[0])]

    def _embed(self, series: torch.Tensor) -> torch.Tensor:
        b, m, t = series.shape
        flat = series.reshape(b * m, t)
        if not self.cache_embeddings:
            return self._assemble(self._encode_series(flat), b, m).to(series.device)
        cache = _EMBED_CACHE.setdefault(self._cache_key(), {})
        keys = self.series_keys(flat)
        missing = list({k: i for i, k in enumerate(keys) if k not in cache}.values())
        if missing:
            fresh = self._encode_series(flat[missing])
            for j, i in enumerate(missing):
                cache[keys[i]] = fresh[j]
        per_series = torch.stack([cache[k] for k in keys], dim=0)
        return self._assemble(per_series, b, m).to(series.device)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_sensor = x[:, :, : len(self.modalities)]
        features = []
        if self.include_embedding:
            self._ensure_model(x.device)
            features.append(self._embed(self.encoder_input(x)))
        if self.append_input_stats:
            features.append(Chronos2EncoderNet.input_stats(x_sensor))
            if self.second_block == "subject" and self.second_cols is not None:
                features.append(Chronos2EncoderNet.input_stats(x[:, :, self.second_cols]))
        if self.append_time_features and self.time_cols is not None:
            features.append(x[:, -1, self.time_cols])
        if demographics is not None:
            features.append(demographics)
        return self.fc(torch.cat(features, dim=-1))


if __name__ == "__main__":
    print(TiRexEncoderNet(modalities=["step", "distance"]))
