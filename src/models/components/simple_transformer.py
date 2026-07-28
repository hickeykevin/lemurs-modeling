import math
import torch
import torch.nn as nn
from typing import Optional


class SimpleTransformer(nn.Module):
    """A lightweight Transformer Encoder model for multimodal sequence classification.

    Processes sequential features with positional encoding and self-attention,
    pools the sequence representation (mean, max, or last), and fuses optional
    static demographics.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 2,
        dropout: float = 0.1,
        use_sequence_data: bool = True,
        demographics_dim: int = 0,
        pooling: str = "mean",
        max_len: int = 500,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.use_sequence_data = use_sequence_data
        self.demographics_dim = demographics_dim
        self.max_len = max_len

        if pooling not in ("last", "mean", "max"):
            raise ValueError(f"pooling must be one of 'last', 'mean', 'max'; got {pooling!r}")
        self.pooling = pooling

        # Sequence processing & Transformer Encoder
        if self.use_sequence_data:
            self.input_size = input_size
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, hidden_size))
            nn.init.normal_(self.pos_encoder, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            fc_input_size = hidden_size
        else:
            fc_input_size = 0

        fc_input_size += self.demographics_dim
        self.fc = nn.Linear(max(1, fc_input_size), output_size)

    def init_input_size(self, input_size: int) -> None:
        """Dynamically adjusts the input linear projection if feature count differs."""
        if self.use_sequence_data and (not hasattr(self, "input_size") or input_size != self.input_size):
            self.input_size = input_size
            self.input_proj = nn.Linear(input_size, self.hidden_size)

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim != demographics_dim:
            self.demographics_dim = demographics_dim
            fc_input_size = self.hidden_size if self.use_sequence_data else 0
            in_features = fc_input_size + demographics_dim
            out_features = self.fc.out_features
            self.fc = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the SimpleTransformer module."""
        if self.use_sequence_data and x is not None:
            if x.shape[-1] != getattr(self, "input_size", x.shape[-1]):
                self.init_input_size(x.shape[-1])

            batch_size, seq_len, _ = x.shape
            out = self.input_proj(x)

            if seq_len > self.max_len:
                pos_emb = self.pos_encoder[:, :self.max_len, :]
                pos_emb = torch.cat([pos_emb, torch.zeros(1, seq_len - self.max_len, self.hidden_size, device=x.device)], dim=1)
            else:
                pos_emb = self.pos_encoder[:, :seq_len, :]

            out = out + pos_emb
            out = self.transformer_encoder(out)

            if self.pooling == "mean":
                last_out = out.mean(dim=1)
            elif self.pooling == "max":
                last_out = out.max(dim=1).values
            else:  # "last"
                last_out = out[:, -1, :]
        else:
            last_out = None

        if demographics is not None:
            if last_out is not None:
                last_out = torch.cat([last_out, demographics], dim=-1)
            else:
                last_out = demographics

        if last_out is None:
            raise ValueError(
                "At least one feature source (use_sequence_data or demographics) must be provided."
            )

        logits = self.fc(last_out)
        return logits


if __name__ == "__main__":
    model = SimpleTransformer()
    x = torch.randn(8, 24, 2)
    logits = model(x)
    print("SimpleTransformer test output shape:", logits.shape)
