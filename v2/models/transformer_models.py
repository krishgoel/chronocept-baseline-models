"""
Transformer-based models: RoBERTa, DeBERTa, DistilBERT with different heads and losses.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import logging

from .base import BaseTransformerModel
from .encoders import RoBERTaEncoder, DeBERTaEncoder, DistilBERTEncoder, AxisEncoder
from .heads import LinearHead, FFNNHead, MultiTaskHead
from .pooling import MeanPooling, AttentionPooling, PoolerOutput
from .losses import SkewNormalNLL, GaussianNLL, MSELoss, ParamSpaceGaussianNLL

logger = logging.getLogger(__name__)


class RoBERTaRegression(BaseTransformerModel):
    """
    RoBERTa-base + linear head (MSE) - standard regression baseline.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 pooling_type: str = "mean",
                 head_type: str = "linear",
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "mse",
                 dropout: float = 0.1,
                 hidden_dim: int = 512):
        params = {
            "model_name": model_name,
            "pooling_type": pooling_type,
            "head_type": head_type,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type,
            "dropout": dropout,
            "hidden_dim": hidden_dim
        }
        super().__init__(params)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        
        # Initialize encoder
        self.encoder = RoBERTaEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Initialize pooling
        if pooling_type == "mean":
            self.pooling = MeanPooling()
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(self.encoder.hidden_size)
        elif pooling_type == "pooler":
            self.pooling = PoolerOutput()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Determine output dimension based on loss type
        head_output_dim = 6 if loss_type == "param_gauss" else 3

        # Initialize head
        if head_type == "linear":
            self.head = LinearHead(self.encoder.hidden_size, head_output_dim, dropout)
        elif head_type == "ffnn":
            self.head = FFNNHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout=dropout)
        elif head_type == "multitask":
            self.head = MultiTaskHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        # Initialize loss
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        elif loss_type == "param_gauss":
            self.loss_fn = ParamSpaceGaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built RoBERTa model with {self.pooling.__class__.__name__} pooling and {self.head.__class__.__name__} head")
    
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
        if isinstance(self.pooling, PoolerOutput):
            pooled = self.pooling(encoded['pooler_output'])
        else:
            pooled = self.pooling(encoded['last_hidden_states'], encoded['attention_mask'])
        
        # Predict
        return self.head(pooled)
    
    def train_model(self, train_loader, valid_loader, epochs: int = 10, lr: float = 1e-5, 
                   warm_start_epochs: int = 3) -> Dict[str, Any]:
        """Train the model with optional warm start."""
        self.to(self.device)
        
        # Warm start training if specified
        if warm_start_epochs > 0:
            self.logger.info(f"Starting warm-start training for {warm_start_epochs} epochs")
            warm_history = self.warm_start_training(train_loader, valid_loader, warm_start_epochs, lr)
        
        # Full training
        optimizer = self.get_optimizer(lr)
        
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
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
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
        
        # Transform predictions to parameter means when using param_gauss
        preds = torch.cat(all_predictions, dim=0)
        targs = torch.cat(all_targets, dim=0)

        if self.loss_type == "param_gauss":
            mu_xi = preds[:, 0]
            mu_logw = preds[:, 2]
            mu_alphat = preds[:, 4]
            alpha_bound = getattr(self.loss_fn, 'alpha_bound', 5.0)
            xi_pred = mu_xi
            omega_pred = torch.exp(mu_logw)
            alpha_pred = alpha_bound * torch.tanh(mu_alphat)
            predictions_np = torch.stack([xi_pred, omega_pred, alpha_pred], dim=1).cpu().numpy()
        else:
            predictions_np = preds.cpu().numpy()

        targets_np = targs.numpy()

        # Compute metrics
        from utils_v2.metrics import evaluate_model_comprehensive
        metrics = evaluate_model_comprehensive(predictions_np, targets_np, 
                                               loss_type=("skew_normal" if self.loss_type=="skew_normal" else ("gaussian" if self.loss_type=="gaussian" else "mse")))
        return metrics


class DeBERTaRegression(BaseTransformerModel):
    """
    DeBERTa-V3-base with linear head (MSE) and skew-normal NLL head.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-base",
                 pooling_type: str = "mean",
                 head_type: str = "linear",
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "skew_normal",
                 dropout: float = 0.1,
                 hidden_dim: int = 512):
        params = {
            "model_name": model_name,
            "pooling_type": pooling_type,
            "head_type": head_type,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type,
            "dropout": dropout,
            "hidden_dim": hidden_dim
        }
        super().__init__(params)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        
        # Initialize encoder
        self.encoder = DeBERTaEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Initialize pooling
        if pooling_type == "mean":
            self.pooling = MeanPooling()
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(self.encoder.hidden_size)
        elif pooling_type == "pooler":
            self.pooling = PoolerOutput()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Determine output dimension based on loss type
        head_output_dim = 6 if loss_type == "param_gauss" else 3

        # Initialize head
        if head_type == "linear":
            self.head = LinearHead(self.encoder.hidden_size, head_output_dim, dropout)
        elif head_type == "ffnn":
            self.head = FFNNHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout=dropout)
        elif head_type == "multitask":
            self.head = MultiTaskHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        # Initialize loss
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        elif loss_type == "param_gauss":
            self.loss_fn = ParamSpaceGaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built DeBERTa model with {self.pooling.__class__.__name__} pooling and {self.head.__class__.__name__} head")
    
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
        if isinstance(self.pooling, PoolerOutput):
            pooled = self.pooling(encoded['pooler_output'])
        else:
            pooled = self.pooling(encoded['last_hidden_states'], encoded['attention_mask'])
        
        # Predict
        return self.head(pooled)
    
    def train_model(self, train_loader, valid_loader, epochs: int = 10, lr: float = 1e-5, 
                   warm_start_epochs: int = 3) -> Dict[str, Any]:
        """Train the model with optional warm start."""
        self.to(self.device)
        
        # Warm start training if specified
        if warm_start_epochs > 0:
            self.logger.info(f"Starting warm-start training for {warm_start_epochs} epochs")
            warm_history = self.warm_start_training(train_loader, valid_loader, warm_start_epochs, lr)
        
        # Full training
        optimizer = self.get_optimizer(lr)
        
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
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
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
        
        preds = torch.cat(all_predictions, dim=0)
        targs = torch.cat(all_targets, dim=0)

        if self.loss_type == "param_gauss":
            mu_xi = preds[:, 0]
            mu_logw = preds[:, 2]
            mu_alphat = preds[:, 4]
            alpha_bound = getattr(self.loss_fn, 'alpha_bound', 5.0)
            xi_pred = mu_xi
            omega_pred = torch.exp(mu_logw)
            alpha_pred = alpha_bound * torch.tanh(mu_alphat)
            predictions_np = torch.stack([xi_pred, omega_pred, alpha_pred], dim=1).cpu().numpy()
        else:
            predictions_np = preds.cpu().numpy()

        targets_np = targs.numpy()

        from utils_v2.metrics import evaluate_model_comprehensive
        metrics = evaluate_model_comprehensive(predictions_np, targets_np, 
                                               loss_type=("skew_normal" if self.loss_type=="skew_normal" else ("gaussian" if self.loss_type=="gaussian" else "mse")))
        return metrics


class DistilBERTRegression(BaseTransformerModel):
    """
    DistilBERT + skew-normal NLL head - efficiency baseline.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 pooling_type: str = "mean",
                 head_type: str = "linear",
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "skew_normal",
                 dropout: float = 0.1,
                 hidden_dim: int = 512):
        params = {
            "model_name": model_name,
            "pooling_type": pooling_type,
            "head_type": head_type,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type,
            "dropout": dropout,
            "hidden_dim": hidden_dim
        }
        super().__init__(params)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        
        # Initialize encoder
        self.encoder = DistilBERTEncoder(model_name)
        self.axis_encoder = AxisEncoder()
        
        # Initialize pooling
        if pooling_type == "mean":
            self.pooling = MeanPooling()
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(self.encoder.hidden_size)
        elif pooling_type == "pooler":
            self.pooling = PoolerOutput()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Determine output dimension based on loss type
        head_output_dim = 6 if loss_type == "param_gauss" else 3

        # Initialize head
        if head_type == "linear":
            self.head = LinearHead(self.encoder.hidden_size, head_output_dim, dropout)
        elif head_type == "ffnn":
            self.head = FFNNHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout=dropout)
        elif head_type == "multitask":
            self.head = MultiTaskHead(self.encoder.hidden_size, hidden_dim, head_output_dim, dropout)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        # Initialize loss
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        elif loss_type == "param_gauss":
            self.loss_fn = ParamSpaceGaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built DistilBERT model with {self.pooling.__class__.__name__} pooling and {self.head.__class__.__name__} head")
    
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
        if isinstance(self.pooling, PoolerOutput):
            pooled = self.pooling(encoded['pooler_output'])
        else:
            pooled = self.pooling(encoded['last_hidden_states'], encoded['attention_mask'])
        
        # Predict
        return self.head(pooled)
    
    def train_model(self, train_loader, valid_loader, epochs: int = 15, lr: float = 2e-5, 
                   warm_start_epochs: int = 3) -> Dict[str, Any]:
        """Train the model with optional warm start."""
        self.to(self.device)
        
        # Warm start training if specified
        if warm_start_epochs > 0:
            self.logger.info(f"Starting warm-start training for {warm_start_epochs} epochs")
            warm_history = self.warm_start_training(train_loader, valid_loader, warm_start_epochs, lr)
        
        # Full training
        optimizer = self.get_optimizer(lr)
        
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
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.load_state_dict(best_model_state)
        
        history = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
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
        
        preds = torch.cat(all_predictions, dim=0)
        targs = torch.cat(all_targets, dim=0)

        if self.loss_type == "param_gauss": #Fixed Eval Bug with DistilBERT
            mu_xi = preds[:, 0]
            mu_logw = preds[:, 2]
            mu_alphat = preds[:, 4]
            alpha_bound = getattr(self.loss_fn, 'alpha_bound', 5.0)
            xi_pred = mu_xi
            omega_pred = torch.exp(mu_logw)
            alpha_pred = alpha_bound * torch.tanh(mu_alphat)
            predictions_np = torch.stack([xi_pred, omega_pred, alpha_pred], dim=1).cpu().numpy()
        else:
            predictions_np = preds.cpu().numpy()

        targets_np = targs.numpy()

        from utils_v2.metrics import evaluate_model_comprehensive
        metrics = evaluate_model_comprehensive(predictions_np, targets_np, 
                                               loss_type=("skew_normal" if self.loss_type=="skew_normal" else ("gaussian" if self.loss_type=="gaussian" else "mse")))
        return metrics
