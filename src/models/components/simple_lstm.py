import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.0,
        use_sequence_data: bool = True,
        demographics_dim: int = 0,
        pooling: str = "last",
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_sequence_data = use_sequence_data
        self.demographics_dim = demographics_dim
        if pooling not in ("last", "mean", "max"):
            raise ValueError(f"pooling must be one of 'last', 'mean', 'max'; got {pooling!r}")
        self.pooling = pooling
        
        # LSTM layer
        if self.use_sequence_data:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            fc_input_size = hidden_size
        else:
            fc_input_size = 0

        # Add demographics_dim to fc_input_size
        fc_input_size += self.demographics_dim

        # Fully connected layer to map hidden states to output classes
        self.fc = nn.Linear(max(1, fc_input_size), output_size)

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            fc_input_size = self.hidden_size if self.use_sequence_data else 0
            in_features = fc_input_size + demographics_dim
            out_features = self.fc.out_features
            self.fc = nn.Linear(in_features, out_features)

    def init_input_size(self, input_size: int) -> None:
        """Dynamically adjusts the LSTM input size if different from the default."""
        if self.use_sequence_data and hasattr(self, "lstm") and input_size != self.lstm.input_size:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.lstm.dropout
            )

    def forward(self, x=None, demographics=None):
        # x shape: [Batch, Time=24, Features=input_size]
        if self.use_sequence_data:
            if x is not None and x.shape[-1] != self.lstm.input_size:
                self.init_input_size(x.shape[-1])
            # Forward pass through LSTM
            out, (hn, cn) = self.lstm(x)
            if self.pooling == "mean":
                last_out = out.mean(dim=1)
            elif self.pooling == "max":
                last_out = out.max(dim=1).values
            else:  # "last"
                last_out = out[:, -1, :]
        else:
            last_out = None

        # Concatenate demographics if present
        if demographics is not None:
            if last_out is not None:
                last_out = torch.cat([last_out, demographics], dim=-1)
            else:
                last_out = demographics

        if last_out is None:
            raise ValueError("At least one of use_sequence_data or demographics must be enabled.")

        # Map to logits
        logits = self.fc(last_out)
        return logits

if __name__ == "__main__":
    _ = SimpleLSTM()
