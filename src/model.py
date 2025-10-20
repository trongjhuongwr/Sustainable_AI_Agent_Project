import torch
import torch.nn as nn
from src.config import Config

class WeatherGRU(nn.Module):
    """A GRU-based model for binary weather prediction."""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int, dropout_rate: float):
        super(WeatherGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_time_step_out = gru_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return self.sigmoid(out)

def initialize_model(config: Config) -> WeatherGRU:
    """Initializes the WeatherGRU model using parameters from the Config object."""
    model = WeatherGRU(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE
    )
    print("GRU model architecture defined and initialized.")
    return model