"""
Prediction heads for different model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LinearHead(nn.Module):
    """Simple linear head for regression."""
    
    def __init__(self, input_dim: int, output_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear(x)


class FFNNHead(nn.Module):
    """Feed-forward neural network head."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 3, 
                 num_layers: int = 2, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "swish":
                layers.append(nn.SiLU())
            
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task head with shared trunk and separate heads for each parameter.
    
    Implements MT-DNN style architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3, 
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        
        # Shared trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU() if activation == "relu" else nn.GELU()
        )
        
        # Separate heads for each parameter
        self.xi_head = nn.Linear(hidden_dim, 1)
        self.omega_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_trunk(x)
        
        xi = self.xi_head(shared_features)
        omega = self.omega_head(shared_features)
        alpha = self.alpha_head(shared_features)
        
        return torch.cat([xi, omega, alpha], dim=1)


class BiLSTMHead(nn.Module):
    """BiLSTM head for sequential processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, 
                 num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_concat = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_concat = h_n[-1, :, :]
        
        h_concat = self.dropout(h_concat)
        return self.output_layer(h_concat)


class ResidualHead(nn.Module):
    """Residual head with skip connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 3, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_proj(x)
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                residual = residual + layer(residual)  # Skip connection
            else:
                residual = layer(residual)
        
        return self.output_layer(residual)
