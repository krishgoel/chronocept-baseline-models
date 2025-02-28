import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from transformers import BertModel
from models.base_model import BaseModel
from typing import Dict, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BertRegressionModel(BaseModel, nn.Module):
    """
    BERT Regression Model that adheres to the BaseModel interface.
    
    This model uses a pretrained BERT to extract pooled representations from multiple text fields (the parent text and 8 axes), concatenates them (resulting in a vector of dimension 9 * hidden_size), applies dropout, and feeds the result into a linear regressor to predict three target values.
    
    Hyperparameters (tunable via grid search):
        - dropout: Dropout probability.
    """
    def __init__(self, dropout: float = 0.1):
        """
        Initializes the BERT Regression model.
        
        Args:
            dropout (float): Dropout probability.
        """
        params = {"dropout": dropout}
        super().__init__(params)
        nn.Module.__init__(self)
        self.dropout_rate = dropout
        self.build_model()
    
    def build_model(self):
        """
        Constructs the model:
          - Loads a pretrained BERT model.
          - Initializes a dropout layer.
          - Creates a linear regressor mapping from 9 * hidden_size (pooler output) to 3 targets.
        """
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(self.dropout_rate)
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden_size * 9, 3)
        logger.info(f"BERT Regression model built with dropout={self.dropout_rate} and regressor input dim={hidden_size * 9}")
    
    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Tensor of shape [n_features, batch_size, seq_len].
            attention_masks (torch.Tensor): Tensor of shape [n_features, batch_size, seq_len].
        
        Returns:
            Tensor of shape [batch_size, 3] with predictions.
        """
        n_features = input_ids.shape[0]
        pooled_outputs = []

        device = next(self.bert.parameters()).device
        for i in range(n_features):
            outputs = self.bert(
                input_ids=input_ids[i].to(device),
                attention_mask=attention_masks[i].to(device)
            )
            pooled_outputs.append(outputs.pooler_output)

        cat = torch.cat(pooled_outputs, dim=1)
        cat = self.dropout(cat)
        return self.regressor(cat)
    
    def fit(self, train_loader, valid_loader, epochs: int = 5, lr: float = 1e-4, device: torch.device = torch.device("cpu")) -> Tuple[list, list]:
        """
        Trains the BERT Regression model.
        
        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            valid_loader (DataLoader): PyTorch DataLoader for validation data.
            epochs (int): Number of epochs.
            lr (float): Learning rate.
            device (torch.device): Device to use.
        
        Returns:
            Tuple (train_losses, valid_losses) containing loss histories.
        """
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_losses_all = []
        valid_losses_all = []
        
        for epoch in range(epochs):
            self.train()
            epoch_train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self.forward(batch["input_ids"], batch["attention_masks"])
                loss = loss_fn(preds, batch["targets"].to(device))
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            train_loss_avg = np.mean(epoch_train_losses)
            train_losses_all.append(train_loss_avg)
            
            self.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch in valid_loader:
                    preds = self.forward(batch["input_ids"], batch["attention_masks"])
                    loss = loss_fn(preds, batch["targets"].to(device))
                    epoch_val_losses.append(loss.item())
            val_loss_avg = np.mean(epoch_val_losses)
            valid_losses_all.append(val_loss_avg)
            
            logger.info(f"BERT Regressor Epoch {epoch+1}/{epochs} - Train Loss: {train_loss_avg:.4f} - Val Loss: {val_loss_avg:.4f}")
        
        return train_losses_all, valid_losses_all
    
    def train(self, *args, **kwargs):
        """
        Overloaded train method.
        
        - If called with no arguments or a single boolean (e.g., self.train(False)), toggles the training mode via nn.Module.train.
        - Otherwise, calls fit() and returns the loss histories.
        """
        if (not args and not kwargs) or (len(args) == 1 and isinstance(args[0], bool)):
            return nn.Module.train(self, *args, **kwargs)
        return self.fit(*args, **kwargs)
    
    def evaluate(self, test_loader, device: torch.device = torch.device("cpu")) -> np.ndarray:
        """
        Evaluates the model on test data.
        
        Args:
            test_loader (DataLoader): DataLoader for test data.
            device (torch.device): Device to use.
        
        Returns:
            NumPy array of predictions.
        """
        self.to(device)
        self.eval()
        preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                preds = self.forward(batch["input_ids"], batch["attention_masks"])
                preds_list.append(preds.cpu().numpy())
        return np.vstack(preds_list)
    
    def grid_search(self, train_loader, valid_loader, grid_params: Dict, epochs: int, lr: float, device: torch.device) -> Tuple['BertRegressionModel', Dict, list, list]:
        """
        Performs grid search over hyperparameters.
        
        Expected keys in grid_params:
            - 'dropout': list of dropout rates.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            grid_params (dict): Dictionary of grid search parameters, e.g. {'dropout': [0.1, 0.2, 0.3]}.
            epochs (int): Number of epochs per configuration.
            lr (float): Learning rate.
            device (torch.device): Device to use.
        
        Returns:
            Tuple (best_model, best_params, best_train_losses, best_valid_losses).
        """
        best_val_loss = float("inf")
        best_model = None
        best_params = {}
        best_train_losses = []
        best_valid_losses = []
        
        for dropout in grid_params.get("dropout", []):
            logger.info(f"Grid search: dropout={dropout}, lr={lr}")
            model = BertRegressionModel(dropout=dropout)
            train_losses, valid_losses = model.fit(train_loader, valid_loader, epochs=epochs, lr=lr, device=device)
            final_val_loss = valid_losses[-1]
            logger.info(f"Final validation loss: {final_val_loss:.4f} for dropout={dropout}, lr={lr}")
            
            plt.figure()
            plt.plot(valid_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"BERT Regressor Config: dropout={dropout}, lr={lr}")
            plt.legend()
            plt.show()
            
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model = model
                best_params = {"dropout": dropout, "lr": lr}
                best_train_losses = train_losses
                best_valid_losses = valid_losses
        
        return best_model, best_params, best_train_losses, best_valid_losses
    
    def save(self, filepath: str):
        """
        Saves the model state and hyperparameters to the specified filepath.
        
        Args:
            filepath (str): Path to save the model.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hyperparameters': self.params
        }
        torch.save(checkpoint, filepath)
        logger.info(f"BERT Regressor model saved to {filepath}")
    
    def load(self, filepath: str) -> Dict:
        """
        Loads the model state from the specified filepath.
        
        Args:
            filepath (str): Path from which to load the model.
        
        Returns:
            A dictionary of hyperparameters from the checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"BERT Regressor model loaded from {filepath}")
        return checkpoint.get("hyperparameters", {})