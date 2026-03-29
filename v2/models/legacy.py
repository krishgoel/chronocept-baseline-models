"""
Legacy models for ablation studies and comparisons.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import logging

from .base import BaseTransformerModel
from .encoders import BERTEncoder, AxisEncoder
from .pooling import PoolerOutput
from .heads import LinearHead
from .losses import MSELoss

logger = logging.getLogger(__name__)


class LegacyBERTRegression(BaseTransformerModel):
    """
    Legacy BERT regression with 9-pass concatenation - negative ablation.
    
    This implements the problematic approach mentioned in the review:
    - 9 separate forward passes (parent + 8 axes)
    - Concatenation of pooled outputs
    - MSE loss
    - Pooler output only
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 dropout: float = 0.1):
        params = {
            "model_name": model_name,
            "dropout": dropout,
            "axis_encoding": "9_pass_concat",
            "loss_type": "mse",
            "pooling_type": "pooler"
        }
        super().__init__(params)
        
        # Initialize encoder
        self.encoder = BERTEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Use pooler output (the problematic approach)
        self.pooling = PoolerOutput()
        
        # Linear head for 9 * hidden_size input
        self.head = LinearHead(self.encoder.hidden_size * 9, 3, dropout)
        
        # MSE loss (the problematic loss function)
        self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built Legacy BERT model with 9-pass concatenation (negative ablation)")
    
    def forward(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> torch.Tensor:
        """
        Forward pass with 9-pass concatenation (the problematic approach).
        
        This is exactly what the review criticizes:
        1. 9 separate BERT forward passes
        2. Concatenation of pooled outputs
        3. No axis interaction until final concatenation
        """
        batch_size = len(texts)
        device = next(self.parameters()).device
        
        # Prepare all texts (parent + 8 axes)
        all_texts = []
        
        # Parent text
        all_texts.extend(texts)
        
        # 8 axes texts
        axes_order = [
            "main_outcome_axis", "static_axis", "generic_axis", 
            "hypothetical_axis", "negation_axis", "intention_axis", 
            "opinion_axis", "recurrent_axis"
        ]
        
        for sample_axes in axes_data:
            for axis_key in axes_order:
                axis_text = sample_axes.get(axis_key, "")
                all_texts.append(axis_text)
        
        # Encode all texts in one batch (but conceptually 9 separate passes)
        encoded = self.encoder.encode_texts(all_texts)
        pooler_outputs = encoded['pooler_output']  # [batch_size * 9, hidden_size]
        
        # Reshape to [batch_size, 9, hidden_size]
        pooler_outputs = pooler_outputs.view(batch_size, 9, self.encoder.hidden_size)
        
        # Concatenate along the feature dimension: [batch_size, 9 * hidden_size]
        concatenated = pooler_outputs.view(batch_size, -1)
        
        # Apply dropout and predict
        concatenated = self.head.dropout(concatenated)
        predictions = self.head.linear(concatenated)
        
        return predictions
    
    def train_model(self, train_loader, valid_loader, epochs: int = 5, lr: float = 1e-4) -> Dict[str, Any]:
        """Train the legacy model."""
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        train_losses = []
        valid_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets'].to(self.device)
                
                predictions = self.forward(texts, axes_data)
                loss = self.loss_fn(predictions, targets)
                
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            
            train_loss = torch.tensor(epoch_train_losses).mean().item()
            train_losses.append(train_loss)
            
            # Validation
            self.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    texts = batch['texts']
                    axes_data = batch.get('axes_data', None)
                    targets = batch['targets'].to(self.device)
                    
                    predictions = self.forward(texts, axes_data)
                    loss = self.loss_fn(predictions, targets)
                    epoch_val_losses.append(loss.item())
            
            val_loss = torch.tensor(epoch_val_losses).mean().item()
            valid_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
            
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        """Evaluate the legacy model."""
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets']
                
                predictions = self.forward(texts, axes_data)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets)
        
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Compute metrics
        from utils.metrics import evaluate_model
        mse, mae, r2, nll, crps, pearson, spearman = evaluate_model(targets, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'nll': nll,
            'crps': crps,
            'pearson': pearson,
            'spearman': spearman
        }


class LegacyBERTRegressionSeparatePasses(BaseTransformerModel):
    """
    Legacy BERT regression with truly separate 9 forward passes.
    
    This is the most computationally expensive version that does exactly
    9 separate BERT forward passes as mentioned in the review.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 dropout: float = 0.1):
        params = {
            "model_name": model_name,
            "dropout": dropout,
            "axis_encoding": "9_separate_passes",
            "loss_type": "mse",
            "pooling_type": "pooler"
        }
        super().__init__(params)
        
        # Initialize encoder
        self.encoder = BERTEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Use pooler output
        self.pooling = PoolerOutput()
        
        # Linear head for 9 * hidden_size input
        self.head = LinearHead(self.encoder.hidden_size * 9, 3, dropout)
        
        # MSE loss
        self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built Legacy BERT model with 9 separate forward passes (most expensive)")
    
    def forward(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> torch.Tensor:
        """
        Forward pass with truly separate 9 BERT forward passes.
        
        This is the most computationally expensive approach that does exactly
        what the review criticizes - 9 separate forward passes.
        """
        batch_size = len(texts)
        device = next(self.parameters()).device
        
        pooled_outputs = []
        
        # 9 separate forward passes (the problematic approach)
        for i in range(batch_size):
            sample_pooled = []
            
            # Parent text
            parent_encoded = self.encoder.encode_texts([texts[i]])
            parent_pooled = parent_encoded['pooler_output']
            sample_pooled.append(parent_pooled)
            
            # 8 axes texts
            axes_order = [
                "main_outcome_axis", "static_axis", "generic_axis", 
                "hypothetical_axis", "negation_axis", "intention_axis", 
                "opinion_axis", "recurrent_axis"
            ]
            
            sample_axes = axes_data[i] if axes_data else {}
            for axis_key in axes_order:
                axis_text = sample_axes.get(axis_key, "")
                axis_encoded = self.encoder.encode_texts([axis_text])
                axis_pooled = axis_encoded['pooler_output']
                sample_pooled.append(axis_pooled)
            
            # Concatenate for this sample
            sample_concat = torch.cat(sample_pooled, dim=1)  # [1, 9 * hidden_size]
            pooled_outputs.append(sample_concat)
        
        # Stack all samples
        concatenated = torch.cat(pooled_outputs, dim=0)  # [batch_size, 9 * hidden_size]
        
        # Apply dropout and predict
        concatenated = self.head.dropout(concatenated)
        predictions = self.head.linear(concatenated)
        
        return predictions
    
    def train_model(self, train_loader, valid_loader, epochs: int = 5, lr: float = 1e-4) -> Dict[str, Any]:
        """Train the legacy model with separate passes."""
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        train_losses = []
        valid_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets'].to(self.device)
                
                predictions = self.forward(texts, axes_data)
                loss = self.loss_fn(predictions, targets)
                
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            
            train_loss = torch.tensor(epoch_train_losses).mean().item()
            train_losses.append(train_loss)
            
            # Validation
            self.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    texts = batch['texts']
                    axes_data = batch.get('axes_data', None)
                    targets = batch['targets'].to(self.device)
                    
                    predictions = self.forward(texts, axes_data)
                    loss = self.loss_fn(predictions, targets)
                    epoch_val_losses.append(loss.item())
            
            val_loss = torch.tensor(epoch_val_losses).mean().item()
            valid_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
            
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        """Evaluate the legacy model."""
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets']
                
                predictions = self.forward(texts, axes_data)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets)
        
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Compute metrics
        from utils.metrics import evaluate_model
        mse, mae, r2, nll, crps, pearson, spearman = evaluate_model(targets, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'nll': nll,
            'crps': crps,
            'pearson': pearson,
            'spearman': spearman
        }
