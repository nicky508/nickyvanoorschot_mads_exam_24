from typing import Dict

import gin
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Encoder is 2 separate layers of the LSTM RNN 
class Encoder(nn.Module):
    def __init__(self, seq_len=192, n_features=1, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
    
    # Initializing the hidden numbers of layers
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.view(-1, self.n_features, self.embedding_dim)



class Decoder(nn.Module):
    def __init__(self, seq_len=192, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Calculate hidden dimension
        self.hidden_dim = 2 * input_dim
        
        # Define LSTM layers
        self.rnn1 = nn.LSTM(
            input_size=input_dim,  # Ensure input_size matches input_dim
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure the input shape is [batch_size, 1, input_dim]
        if x.shape[1] != 1:
            raise ValueError(f"Expected input shape [batch_size, 1, input_dim], but got {x.shape}")
        
        # Repeat the input to expand sequence length to seq_len
        x = x.repeat(1, self.seq_len, 1)  # Shape: [batch_size, seq_len, input_dim]
        
        # Pass through first LSTM layer
        x, _ = self.rnn1(x)
        
        # Pass through second LSTM layer
        x, _ = self.rnn2(x)
        
        # Apply output layer
        x = self.output_layer(x)
        
        # Reshape output to match the input shape (batch_size, seq_len, n_features)
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


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len=192, n_features=1, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x