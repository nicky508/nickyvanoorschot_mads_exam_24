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
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(35, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return latent


@gin.configurable
class Decoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 35),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Unflatten(1, (1, 7, 5)),
            nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
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