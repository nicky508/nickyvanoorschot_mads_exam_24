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
            nn.Linear(config["input"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["h3"]),
            nn.ReLU(),
            nn.Linear(config["h3"], config["h4"]),
            nn.ReLU(),
            nn.Linear(config["h4"], config["latent"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return latent


@gin.configurable
class Decoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(config["latent"], config["h4"]),
            nn.ReLU(),
            nn.Linear(config["h4"], config["h3"]),
            nn.ReLU(),
            nn.Linear(config["h3"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["input"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decode(x)
        # x = x.reshape((-1, 28, 28, 1))
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
        
        # Sum over appropriate dimensions
        summed = torch.sum(sqe, dim=1)  # Sum over sequence_length dimension
        
        # Compute mean loss
        mean_loss = torch.mean(summed)
        
        return mean_loss


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