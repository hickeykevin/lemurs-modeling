import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 5, dropout: float = 0.0, user_embedding_dim: int = 16):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.user_embedding_dim = user_embedding_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # Fully connected layer to map hidden states to output classes
        self.fc = nn.Linear(hidden_size, output_size)
        
    def init_user_embedding(self, num_users: int) -> None:
        """Initializes user embedding layer and adjusts the output linear projection."""
        if num_users > 0:
            self.num_users = num_users
            self.user_embedding = nn.Embedding(num_users, self.user_embedding_dim)
            self.fc = nn.Linear(self.hidden_size + self.user_embedding_dim, self.fc.out_features)

    def forward(self, x, user_idx=None):
        # x shape: [Batch, Time=24, Features=input_size]
        
        # Forward pass through LSTM
        # out: [Batch, Time=24, hidden_size]
        out, (hn, cn) = self.lstm(x)
        
        # We take the output from the last time step
        last_out = out[:, -1, :]
        
        # Concatenate user embedding if present and user_idx is provided
        if user_idx is not None and hasattr(self, "user_embedding"):
            embed = self.user_embedding(user_idx)  # [Batch, user_embedding_dim]
            last_out = torch.cat([last_out, embed], dim=-1)
        
        # Map to logits
        logits = self.fc(last_out)
        return logits

if __name__ == "__main__":
    _ = SimpleLSTM()
