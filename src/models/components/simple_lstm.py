import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 5, dropout: float = 0.0):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # Fully connected layer to map hidden states to output classes
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [Batch, Time=24, Features=input_size]
        
        # Forward pass through LSTM
        # out: [Batch, Time=24, hidden_size]
        out, (hn, cn) = self.lstm(x)
        
        # We take the output from the last time step
        last_out = out[:, -1, :]
        
        # Map to logits
        logits = self.fc(last_out)
        return logits

if __name__ == "__main__":
    _ = SimpleLSTM()
