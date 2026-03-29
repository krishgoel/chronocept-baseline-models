"""
MT-DNN style multi-task learning model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import logging

from .base import BaseTransformerModel
from .encoders import RoBERTaEncoder, AxisEncoder
from .pooling import MeanPooling, AttentionPooling
from .heads import MultiTaskHead
from .losses import SkewNormalNLL, GaussianNLL, MSELoss

from utils_v2.metrics import evaluate_model_comprehensive

logger = logging.getLogger(__name__)


class MTDNNModel(BaseTransformerModel):
    """
    MT-DNN style model with shared encoder and multiple task-specific heads.
    
    Implements multi-task learning where the shared encoder learns general
    representations and task-specific heads specialize for different aspects.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 pooling_type: str = "mean",
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "skew_normal",
                 dropout: float = 0.1,
                 hidden_dim: int = 256,
                 num_tasks: int = 3):
        params = {
            "model_name": model_name,
            "pooling_type": pooling_type,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "num_tasks": num_tasks
        }
        super().__init__(params)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        self.num_tasks = num_tasks
        
        # Initialize encoder
        self.encoder = RoBERTaEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Initialize pooling
        if pooling_type == "mean":
            self.pooling = MeanPooling()
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(self.encoder.hidden_size)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Initialize multi-task head
        self.head = MultiTaskHead(
            input_dim=self.encoder.hidden_size,
            hidden_dim=hidden_dim,
            output_dim=3,
            dropout=dropout
        )
        
        # Initialize loss
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built MT-DNN model with {self.pooling.__class__.__name__} pooling")
    
    def forward(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> torch.Tensor:
        """Forward pass."""
        # Process texts based on axis encoding
        if self.axis_encoding == "no_axes":
            processed_texts = self.axis_encoder.no_axes(texts)
        elif self.axis_encoding == "single_sequence_markers":
            processed_texts = self.axis_encoder.single_sequence_markers(texts, axes_data)
        else:
            raise ValueError(f"Unknown axis encoding: {self.axis_encoding}")
        
        # Encode texts
        encoded = self.encoder.encode_texts(processed_texts)
        
        # Pool representations
        pooled = self.pooling(encoded['last_hidden_states'], encoded['attention_mask'])
        
        # Predict using multi-task head
        return self.head(pooled)
    
    def forward_with_auxiliary_losses(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns predictions and auxiliary losses for multi-task learning.
        """
        # Get main predictions
        predictions = self.forward(texts, axes_data)
        
        # Process texts for auxiliary tasks
        if self.axis_encoding == "no_axes":
            processed_texts = self.axis_encoder.no_axes(texts)
        elif self.axis_encoding == "single_sequence_markers":
            processed_texts = self.axis_encoder.single_sequence_markers(texts, axes_data)
        
        # Encode texts
        encoded = self.encoder.encode_texts(processed_texts)
        pooled = self.pooling(encoded['last_hidden_states'], encoded['attention_mask'])
        
        # Extract shared features from the multi-task head
        shared_features = self.head.shared_trunk(pooled)
        
        # Compute auxiliary losses (e.g., parameter-specific losses)
        auxiliary_losses = {}
        
        # Xi-specific loss (location parameter)
        xi_pred = self.head.xi_head(shared_features)
        xi_loss = torch.mean(torch.abs(xi_pred))  # L1 regularization
        
        # Omega-specific loss (scale parameter)
        omega_pred = self.head.omega_head(shared_features)
        omega_loss = torch.mean(torch.abs(omega_pred))  # L1 regularization
        
        # Alpha-specific loss (shape parameter)
        alpha_pred = self.head.alpha_head(shared_features)
        alpha_loss = torch.mean(torch.abs(alpha_pred))  # L1 regularization
        
        auxiliary_losses = {
            'xi_regularization': xi_loss,
            'omega_regularization': omega_loss,
            'alpha_regularization': alpha_loss
        }
        
        return {
            'predictions': predictions,
            'auxiliary_losses': auxiliary_losses
        }
    
    def train_model(self, train_loader, valid_loader, epochs: int = 15, lr: float = 1e-5, 
                   warm_start_epochs: int = 3, auxiliary_weight: float = 0.1) -> Dict[str, Any]:
        """Train the model with multi-task learning."""
        self.to(self.device)
        
        # Warm start training if specified
        if warm_start_epochs > 0:
            self.logger.info(f"Starting warm-start training for {warm_start_epochs} epochs")
            warm_history = self.warm_start_training(train_loader, valid_loader, warm_start_epochs, lr)
        
        # Full training with multi-task learning
        optimizer = self.get_optimizer(lr)
        
        train_losses = []
        valid_losses = []
        auxiliary_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_train_losses = []
            epoch_aux_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets'].to(self.device)
                
                # Forward pass with auxiliary losses
                outputs = self.forward_with_auxiliary_losses(texts, axes_data)
                predictions = outputs['predictions']
                aux_losses = outputs['auxiliary_losses']
                
                # Main task loss
                main_loss = self.loss_fn(predictions, targets)
                
                # Auxiliary losses
                total_aux_loss = sum(aux_losses.values())
                
                # Combined loss
                total_loss = main_loss + auxiliary_weight * total_aux_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(main_loss.item())
                epoch_aux_losses.append(total_aux_loss.item())
            
            train_loss = torch.tensor(epoch_train_losses).mean().item()
            aux_loss = torch.tensor(epoch_aux_losses).mean().item()
            train_losses.append(train_loss)
            auxiliary_losses.append(aux_loss)
            
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
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Aux Loss: {aux_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'auxiliary_losses': auxiliary_losses,
            'best_val_loss': best_val_loss
        }
        
        if warm_start_epochs > 0:
            history['warm_start'] = warm_history
        
        return history
    
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        """Evaluate the model."""
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
        
        metrics = evaluate_model_comprehensive(predictions, targets, loss_type=("skew_normal" if self.loss_type=="skew_normal" else ("gaussian" if self.loss_type=="gaussian" else "mse")))
        return metrics
