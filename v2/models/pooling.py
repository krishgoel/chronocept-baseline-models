"""
Pooling strategies for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MeanPooling(nn.Module):
    """Mean pooling over sequence length."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        # Mask out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class AttentionPooling(nn.Module):
    """Learned attention pooling."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        # Compute attention weights
        attention_weights = self.attention(last_hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask out padding tokens
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(last_hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return pooled


class PoolerOutput(nn.Module):
    """Use the pooler output from transformer (e.g., BERT's [CLS] token)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pooler_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pooler_output: [batch_size, hidden_size] - already pooled
            attention_mask: Not used, kept for interface compatibility
        """
        return pooler_output


class MaxPooling(nn.Module):
    """Max pooling over sequence length."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        # Mask out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        last_hidden_states = last_hidden_states.masked_fill(input_mask_expanded == 0, -1e9)
        return torch.max(last_hidden_states, dim=1)[0]


class CLSPooling(nn.Module):
    """Use only the [CLS] token (first token)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, last_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            last_hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Not used, kept for interface compatibility
        """
        return last_hidden_states[:, 0, :]  # [CLS] token
