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
        user_embedding_dim: int = 16,
        use_user_embedding: bool = True,
        use_sequence_data: bool = True,
        use_prev_prediction: bool = False,
        demographics_dim: int = 0
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.user_embedding_dim = user_embedding_dim
        self.use_user_embedding = use_user_embedding
        self.use_sequence_data = use_sequence_data
        self.use_prev_prediction = use_prev_prediction
        self.demographics_dim = demographics_dim
        
        # LSTM layer
        if self.use_sequence_data:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            fc_input_size = hidden_size
        else:
            fc_input_size = 0
            
        if self.use_prev_prediction:
            fc_input_size += 1
            
        # Add demographics_dim to fc_input_size
        fc_input_size += self.demographics_dim
            
        # Fully connected layer to map hidden states to output classes
        self.fc = nn.Linear(max(1, fc_input_size), output_size)
        
    def init_user_embedding(self, num_users: int) -> None:
        """Initializes user embedding layer and adjusts the output linear projection."""
        if not self.use_user_embedding:
            return
            
        if num_users > 0:
            self.num_users = num_users
            self.user_embedding = nn.Embedding(num_users, self.user_embedding_dim)
            
            fc_input_size = 0
            if self.use_sequence_data:
                fc_input_size += self.hidden_size
            if self.use_prev_prediction:
                fc_input_size += 1
                
            self.fc = nn.Linear(fc_input_size + self.user_embedding_dim + self.demographics_dim, self.fc.out_features)

    def init_demographics(self, demographics_dim: int) -> None:
        """Adjusts the output linear projection to support static demographics."""
        if demographics_dim > 0 and self.demographics_dim == 0:
            self.demographics_dim = demographics_dim
            # Recalculate input features from scratch to avoid dummy size-0 guard mismatches
            fc_input_size = 0
            if self.use_sequence_data:
                fc_input_size += self.hidden_size
            if self.use_prev_prediction:
                fc_input_size += 1
            if self.use_user_embedding and hasattr(self, "user_embedding"):
                fc_input_size += self.user_embedding_dim
                
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

    def forward(self, x, user_idx=None, prev_pred=None, demographics=None):
        # x shape: [Batch, Time=24, Features=input_size]
        if self.use_sequence_data:
            if x is not None and x.shape[-1] != self.lstm.input_size:
                self.init_input_size(x.shape[-1], device=x.device)
            # Forward pass through LSTM
            # out: [Batch, Time=24, hidden_size]
            out, (hn, cn) = self.lstm(x)
            # We take the output from the last time step
            last_out = out[:, -1, :]
        else:
            last_out = None
        
        # Concatenate prev_pred if enabled
        if self.use_prev_prediction:
            if prev_pred is None:
                # Fallback default
                batch_size = x.shape[0] if x is not None else (user_idx.shape[0] if user_idx is not None else 1)
                prev_pred = torch.zeros(batch_size, dtype=torch.float, device=x.device if x is not None else (user_idx.device if user_idx is not None else None))
            
            if prev_pred.dim() == 1:
                prev_pred = prev_pred.unsqueeze(-1)
                
            if last_out is not None:
                last_out = torch.cat([last_out, prev_pred], dim=-1)
            else:
                last_out = prev_pred
        
        # Concatenate user embedding if present and user_idx is provided
        if self.use_user_embedding and user_idx is not None and hasattr(self, "user_embedding"):
            embed = self.user_embedding(user_idx)  # [Batch, user_embedding_dim]
            if last_out is not None:
                last_out = torch.cat([last_out, embed], dim=-1)
            else:
                last_out = embed
                
        # Concatenate demographics if present
        if demographics is not None:
            if last_out is not None:
                last_out = torch.cat([last_out, demographics], dim=-1)
            else:
                last_out = demographics
                
        if last_out is None:
            raise ValueError("At least one of use_sequence_data, use_user_embedding, use_prev_prediction, or demographics must be enabled.")
        
        # Map to logits
        logits = self.fc(last_out)
        return logits

if __name__ == "__main__":
    _ = SimpleLSTM()
