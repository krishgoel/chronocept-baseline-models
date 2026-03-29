"""
SBERT-based models: SBERT + FFNN and SBERT + BiLSTM.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import logging

from .base import BaseModel
from .encoders import SBERTEncoder, AxisEncoder
from .heads import FFNNHead, BiLSTMHead
from .losses import SkewNormalNLL, GaussianNLL, MSELoss

from utils_v2.metrics import evaluate_model_comprehensive

logger = logging.getLogger(__name__)


class SBERTFFNN(BaseModel, nn.Module):
    """
    SBERT + FFNN baseline - our SoTA embedding baseline.
    
    Uses Sentence-BERT for encoding and feed-forward network for prediction.
    """
    
    def __init__(self, 
                 sbert_model: str = "all-MiniLM-L6-v2",
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "skew_normal"):
        params = {
            "sbert_model": sbert_model,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type
        }
        super().__init__(params)
        nn.Module.__init__(self)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        
        # Initialize components
        self.encoder = SBERTEncoder(sbert_model)
        self.axis_encoder = AxisEncoder()
        
        # Determine input dimension based on axis encoding
        if axis_encoding == "sbert_concat":
            input_dim = self.encoder.hidden_size * 9  # parent + 8 axes
        else:
            input_dim = self.encoder.hidden_size
        
        self.head = FFNNHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Loss function
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built SBERT+FFNN model with {self.axis_encoding} axis encoding")
    
    def forward(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> torch.Tensor:
        """Forward pass."""
        if self.axis_encoding == "no_axes":
            processed_texts = self.axis_encoder.no_axes(texts)
            embeddings = self.encoder.encode_texts(processed_texts)['embeddings']
        elif self.axis_encoding == "single_sequence_markers":
            processed_texts = self.axis_encoder.single_sequence_markers(texts, axes_data)
            embeddings = self.encoder.encode_texts(processed_texts)['embeddings']
        elif self.axis_encoding == "sbert_concat":
            embeddings = self.axis_encoder.sbert_concat(texts, axes_data, self.encoder)
        else:
            raise ValueError(f"Unknown axis encoding: {self.axis_encoding}")
        # Ensure embeddings are normal tensors (not inference tensors) for autograd safety
        embeddings = embeddings.clone().detach()
        
        return self.head(embeddings)
    
    def train_model(self, train_loader, valid_loader, epochs: int = 50, lr: float = 1e-3) -> Dict[str, Any]:
        """Train the model."""
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
            
            if epoch % 10 == 0:
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
        
        return evaluate_model_comprehensive(predictions, targets, loss_type="skew_normal")


class SBERTBiLSTM(BaseModel, nn.Module):
    """
    SBERT + BiLSTM baseline.
    
    Uses Sentence-BERT for encoding and BiLSTM for sequential processing.
    """
    
    def __init__(self, 
                 sbert_model: str = "all-MiniLM-L6-v2",
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 axis_encoding: str = "single_sequence_markers",
                 loss_type: str = "skew_normal"):
        params = {
            "sbert_model": sbert_model,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "axis_encoding": axis_encoding,
            "loss_type": loss_type
        }
        super().__init__(params)
        nn.Module.__init__(self)
        
        self.axis_encoding = axis_encoding
        self.loss_type = loss_type
        
        # Initialize components
        self.encoder = SBERTEncoder(sbert_model)
        self.axis_encoder = AxisEncoder()
        
        # For BiLSTM, we need sequential data
        # We'll create sequences by splitting text into chunks
        self.head = BiLSTMHead(
            input_dim=self.encoder.hidden_size,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Loss function
        if loss_type == "skew_normal":
            self.loss_fn = SkewNormalNLL()
        elif loss_type == "gaussian":
            self.loss_fn = GaussianNLL()
        else:
            self.loss_fn = MSELoss()
        
        self.build_model()
    
    def build_model(self) -> None:
        """Build the complete model."""
        self.logger.info(f"Built SBERT+BiLSTM model with {self.axis_encoding} axis encoding")
    
    def _create_sequences(self, embeddings: torch.Tensor, seq_length: int = 10) -> torch.Tensor:
        """Create sequences from embeddings for BiLSTM processing."""
        batch_size, emb_dim = embeddings.shape
        
        # For simplicity, we'll repeat the embedding to create a sequence
        # In practice, you might want to split text into chunks and encode each chunk
        sequences = embeddings.unsqueeze(1).repeat(1, seq_length, 1)
        return sequences
    
    def forward(self, texts: List[str], axes_data: List[Dict[str, str]] = None) -> torch.Tensor:
        """Forward pass."""
        if self.axis_encoding == "no_axes":
            processed_texts = self.axis_encoder.no_axes(texts)
        elif self.axis_encoding == "single_sequence_markers":
            processed_texts = self.axis_encoder.single_sequence_markers(texts, axes_data)
        else:
            raise ValueError(f"Unknown axis encoding: {self.axis_encoding}")
        
        embeddings = self.encoder.encode_texts(processed_texts)['embeddings']
        sequences = self._create_sequences(embeddings)
        return self.head(sequences)
    
    def train_model(self, train_loader, valid_loader, epochs: int = 50, lr: float = 1e-3) -> Dict[str, Any]:
        """Train the model."""
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
            
            if epoch % 10 == 0:
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
        
        return evaluate_model_comprehensive(predictions, targets, loss_type="skew_normal")
