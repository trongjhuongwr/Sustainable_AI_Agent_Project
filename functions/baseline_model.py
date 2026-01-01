import torch
import torch.nn as nn

class WeatherGRU(nn.Module):
    """
    --- Model Architecture (Baseline GRU)---
    Standard Gated Recurrent Unit (GRU) architecture for time-series binary classification.
    Constructed to serve as a baseline for subsequent compression experiments.
    """
    def __init__(self, config):
        super(WeatherGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.N_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.N_LAYERS > 1 else 0
        )
        self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        
        # Utilize the hidden state from the last time step
        # out[:, -1, :] shape: (batch_size, hidden_dim)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
