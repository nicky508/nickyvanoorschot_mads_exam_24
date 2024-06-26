from typing import Dict, Optional

import gin
import numpy as np
import torch
from pathlib import Path
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@gin.configurable
class Encoder(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super(Encoder, self).__init__()
        if config is None:
            config = {
                "seq_len": gin.REQUIRED,
                "features": gin.REQUIRED,
                "latent": gin.REQUIRED,
                "hidden": gin.REQUIRED,
                "num_layers": gin.REQUIRED,
                "dropout": gin.REQUIRED
            }
        self.seq_len = config["seq_len"]
        self.n_features = config["features"]
        self.latent = config["latent"]
        self.hidden = config["hidden"]
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden,
            num_layers=config["num_layers"],
            batch_first=True
        )
    
        self.rnn2 = nn.LSTM(
            input_size=self.hidden,
            hidden_size=self.latent,
            num_layers=config["num_layers"],
            batch_first=True
        )
        
        self.dropout = nn.Dropout(p=config["dropout"])
        
        # Update the input dimension for the linear layer
        self.fc = nn.Linear(self.seq_len * self.latent, self.latent)
        
    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, (hidden_n, _) = self.rnn2(x)
        x = self.dropout(x)
        batch_size = x.size(0)  # Get the batch size from input x
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, seq_len * embedding_dim)
        x = self.fc(x)
        return x

@gin.configurable
class Decoder(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super(Decoder, self).__init__()
        if config is None:
            config = {
                "seq_len": gin.REQUIRED,
                "features": gin.REQUIRED,
                "latent": gin.REQUIRED,
                "hidden": gin.REQUIRED,
                "num_layers": gin.REQUIRED,
                "dropout": gin.REQUIRED
            }
        self.seq_len = self.seq_len = config["seq_len"]
        self.latent = config["latent"]
        self.hidden = config["hidden"]
        self.n_features = config["features"]

        self.fc = nn.Linear(self.latent, self.seq_len * self.latent)
        
        self.rnn1 = nn.LSTM(
            input_size=self.latent,  # Ensure input_size matches input_dim
            hidden_size=self.hidden,
            num_layers=config["num_layers"],
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=self.hidden,  # Ensure input_size matches input_dim
            hidden_size=self.hidden,
            num_layers=config["num_layers"],
            batch_first=True
        )
        
        self.dropout = nn.Dropout(p=config["dropout"])
        
        self.output_layer = nn.Linear(self.hidden, self.n_features)
        
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from input x
        x = self.fc(x)  # Reverse the effect of encoder's fc
        x = x.view(batch_size, self.seq_len, self.latent)  # Reshape to match LSTM input
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, encoder_config: Optional[Dict] = None, decoder_config: Optional[Dict] = None,):
        super(RecurrentAutoencoder, self).__init__()
        
        self.encoder = Encoder(encoder_config).to(device)
        self.decoder = Decoder(decoder_config).to(device)
        # self.encoder = Encoder(n_features, embedding_dim).to(device)
        # self.decoder = Decoder(embedding_dim, n_features).to(device)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
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