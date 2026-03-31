"""
Base classes for improved baseline models.
"""

import abc
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
from datetime import datetime


class BaseModel(abc.ABC):
    """Base class for all models with common functionality."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.training_history = {}
        self.best_metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abc.abstractmethod
    def build_model(self) -> None:
        """Construct the model architecture."""
        pass
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abc.abstractmethod
    def train_model(self, train_loader, valid_loader, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abc.abstractmethod
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        pass
    
    def save_checkpoint(self, filepath: Union[str, Path], epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict() if hasattr(self, 'model') else self.state_dict(),
            'params': self.params,
            'epoch': epoch,
            'metrics': metrics,
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint['model_state_dict'])
        
        self.params = checkpoint.get('params', self.params)
        self.training_history = checkpoint.get('training_history', {})
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics with proper formatting."""
        for key, value in metrics.items():
            self.logger.info(f"{prefix}{key}: {value:.6f}")


class BaseTransformerModel(BaseModel, nn.Module):
    """Base class for transformer-based models."""
    
    def __init__(self, params: Dict[str, Any]):
        BaseModel.__init__(self, params)
        nn.Module.__init__(self)
        self.encoder = None
        self.pooling = None
        self.head = None
        self.loss_fn = None
        
    def to(self, device: torch.device) -> 'BaseTransformerModel':
        """Move model to device."""
        self.device = device
        return super().to(device)
    
    def get_optimizer(self, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """Get optimizer with layer-wise learning rate decay."""
        # Separate parameters for encoder and head
        encoder_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if 'encoder' in name or 'pooling' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        
        # Use different learning rates for encoder and head
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr * 5, 'weight_decay': weight_decay}  # 5x LR for head
        ])
        
        return optimizer
    
    def warm_start_training(self, train_loader, valid_loader, epochs: int = 3, lr: float = 1e-4) -> Dict[str, Any]:
        """Warm start training: train only xi parameter first."""
        self.logger.info("Starting warm-start training (xi only)")
        
        # Temporarily modify loss to only use xi
        original_loss_fn = self.loss_fn

        # Create a proper loss module for xi-only training
        class XiOnlyLoss(nn.Module):
            def __init__(self, original_loss_fn):
                super().__init__()
                self.original_loss_fn = original_loss_fn
            
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # If prediction provides param-space Gaussian stats (6 dims), use mu_xi at index 0
                if pred.dim() >= 2 and pred.size(-1) >= 1:
                    if pred.size(-1) == 6:
                        mu_xi = pred[:, 0:1]
                    else:
                        # Assume first dim is xi
                        mu_xi = pred[:, 0:1]
                else:
                    mu_xi = pred
                xi_true = target[:, 0:1]
                return torch.nn.functional.mse_loss(mu_xi, xi_true)

        self.loss_fn = XiOnlyLoss(original_loss_fn)
        
        # Do warm start training directly (avoid recursion)
        self.to(self.device)
        optimizer = self.get_optimizer(lr)
        
        train_losses = []
        valid_losses = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move targets to device
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                predictions = self.forward(batch['texts'], batch['axes_data'])
                loss = self.loss_fn(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    # Move targets to device
                    targets = batch['targets'].to(self.device)
                    
                    predictions = self.forward(batch['texts'], batch['axes_data'])
                    loss = self.loss_fn(predictions, targets)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(valid_loader)
            
            train_losses.append(avg_train_loss)
            valid_losses.append(avg_val_loss)
            
            self.logger.info(f"Warm-start Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Restore original loss function
        self.loss_fn = original_loss_fn
        
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
        
        self.logger.info("Warm-start training completed")
        return history
