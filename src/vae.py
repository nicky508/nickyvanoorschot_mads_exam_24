from typing import Dict

import gin
import numpy as np
import torch
from torch import nn

@gin.configurable
class Encoder(nn.Module):
    """encoder"""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.LSTM(config["input"], config["h1"], num_layers=config["lstm_layers"], batch_first=True),
            # nn.Linear(config["input"], config["h1"]),
            nn.MultiheadAttention(
            embed_dim=config["h1"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True),
            nn.Linear(config["h1"], config["h2"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["h3"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h3"], config["h4"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["latent"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return latent


@gin.configurable
class Decoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(config["latent"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h4"], config["h3"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h3"], config["h2"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["h1"]),
            nn.Dropout(config["dropout"]),
            nn.MultiheadAttention(
            embed_dim=config["h1"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True),
            nn.LSTM(config["input"], config["h1"], num_layers=config["lstm_layers"], batch_first=True),
            # nn.Linear(config["h1"], config["input"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decode(x)
        return x


class ReconstructionLoss:
    def __call__(self, y, yhat):
        # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(yhat, np.ndarray):
            yhat = torch.from_numpy(yhat)
        
        # Compute squared error
        sqe = (y - yhat) ** 2
        

        summed = torch.sum(sqe, dim=1) 
        return torch.mean(summed)

@gin.configurable
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x