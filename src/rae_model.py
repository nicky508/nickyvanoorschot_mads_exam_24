from typing import Dict

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
    
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Update the input dimension for the linear layer
        self.fc = nn.Linear(seq_len * embedding_dim, embedding_dim)
        
    def forward(self, x):
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        batch_size = x.size(0)  # Get the batch size from input x
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, seq_len * embedding_dim)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, seq_len=192, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = input_dim  # Adjusted hidden dimension
        
        self.fc = nn.Linear(input_dim, seq_len * input_dim)
        
        self.rnn1 = nn.LSTM(
            input_size=input_dim,  # Ensure input_size matches input_dim
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(input_dim, n_features)
        
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from input x
        x = self.fc(x)  # Reverse the effect of encoder's fc
        x = x.view(batch_size, self.seq_len, self.input_dim)  # Reshape to match LSTM input
        x, _ = self.rnn1(x)
        x = self.output_layer(x)
        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len=192, n_features=1, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
        
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