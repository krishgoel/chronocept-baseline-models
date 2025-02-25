import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class FFNNModel(BaseModel, nn.Module):
    """
    Feed-Forward Neural Network Model that inherits from BaseModel.
    
    Hyperparameters (to be tuned via grid search):
      - hidden_dim: Size of the first hidden layer.
      - dropout: Dropout probability.
      - weight_decay: L2 regularization factor.
      - l1: L1 regularization factor.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0, weight_decay: float = 0.0, l1: float = 0.0):
        params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "l1": l1
        }
        super().__init__(params)
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.weight_decay = weight_decay  # This is used as L2 regularization in the optimizer
        self.l1 = l1  # L1 regularization factor
        
        self.build_model()

    def build_model(self):
        """Construct the FFNN architecture."""
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        logger.info(f"FFNN built with input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                    f"dropout={self.dropout}, weight_decay={self.weight_decay}, l1={self.l1}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def fit(self, train_data, valid_data, lr: float, epochs: int = 50, 
            device: torch.device = torch.device("cpu"), l1: float = None):
        """
        The actual training loop. Returns (train_losses, valid_losses).
        
        Args:
            train_data: Tuple (X_train, y_train) with NumPy arrays.
            valid_data: Tuple (X_valid, y_valid) with NumPy arrays.
            lr: Learning rate.
            epochs: Number of epochs.
            device: Device for training.
            l1: L1 regularization factor (if None, uses self.l1).
        """
        if l1 is None:
            l1 = self.l1

        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        train_losses = []
        valid_losses = []
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(epochs):
            nn.Module.train(self)
            optimizer.zero_grad()
            X_train, y_train = train_data
            inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
            targets = torch.tensor(y_train, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            
            if l1 > 0.0:
                l1_penalty = sum(param.abs().sum() for param in self.parameters())
                loss = loss + l1 * l1_penalty
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            nn.Module.eval(self)
            with torch.no_grad():
                X_valid, y_valid = valid_data
                val_inputs = torch.tensor(X_valid, dtype=torch.float32).to(device)
                val_targets = torch.tensor(y_valid, dtype=torch.float32).to(device)
                val_outputs = self.forward(val_inputs)
                val_loss = criterion(val_outputs, val_targets).item()
                valid_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
            
        if best_model_state:
            self.load_state_dict(best_model_state)
            logger.info("Loaded best model state based on validation loss.")
        return train_losses, valid_losses

    def train(self, *args, **kwargs):
        """
        Overloaded train method.
        
        - If called with no arguments or a single boolean (e.g., self.train(False)),
          it toggles the training mode by directly invoking nn.Module.train.
        - If called with training data (as in grid search), it calls fit() and returns the losses.
        """
        if (not args and not kwargs) or (len(args) == 1 and isinstance(args[0], bool)):
            return nn.Module.train(self, *args, **kwargs)
        return self.fit(*args, **kwargs)

    def evaluate(self, test_data, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Evaluates the model on test data.
        
        Args:
            test_data: Tuple (X_test, y_test) with NumPy arrays.
            device: Device for evaluation.
        
        Returns:
            Predictions as a NumPy array.
        """
        self.to(device)
        nn.Module.eval(self)
        X_test, _ = test_data
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
        return outputs.cpu().numpy()

    def grid_search(self, train_data, valid_data, grid_params: dict, epochs: int, 
                    device: torch.device):
        """
        Performs grid search over hyperparameters defined in grid_params.
        
        Expected keys in grid_params:
            - 'hidden_dim': list of ints
            - 'dropout': list of floats
            - 'weight_decay': list of floats (L2 regularization)
            - 'lr': list of floats
            - 'l1': list of floats (L1 regularization)
        
        Returns:
            Tuple (best_model, best_params, best_train_losses, best_valid_losses)
        """
        best_val_loss = float("inf")
        best_model = None
        best_params = {}
        best_train_losses = []
        best_valid_losses = []

        for hd in grid_params.get("hidden_dim", []):
            for dropout in grid_params.get("dropout", []):
                for wd in grid_params.get("weight_decay", []):
                    for lr in grid_params.get("lr", []):
                        for l1 in grid_params.get("l1", [0.0]):
                            logger.info(f"Grid search: hidden_dim={hd}, dropout={dropout}, "
                                        f"weight_decay={wd}, lr={lr}, l1={l1}")
                            model = FFNNModel(self.input_dim, hidden_dim=hd, dropout=dropout, 
                                                weight_decay=wd, l1=l1)
                            train_losses, valid_losses = model.train(
                                train_data, valid_data, lr=lr, epochs=epochs, device=device, l1=l1
                            )
                            final_val_loss = valid_losses[-1]
                            logger.info(f"Final validation loss for configuration: {final_val_loss:.4f}")

                            plt.figure()
                            plt.plot(valid_losses, label="Validation Loss")
                            plt.xlabel("Epoch")
                            plt.ylabel("Loss")
                            plt.title(f"Config: hd={hd}, dropout={dropout}, wd={wd}, lr={lr}, l1={l1}")
                            plt.legend()
                            plt.show()

                            if final_val_loss < best_val_loss:
                                best_val_loss = final_val_loss
                                best_model = model
                                best_params = {
                                    "hidden_dim": hd,
                                    "dropout": dropout,
                                    "weight_decay": wd,
                                    "lr": lr,
                                    "l1": l1
                                }
                                best_train_losses = train_losses
                                best_valid_losses = valid_losses

        return best_model, best_params, best_train_losses, best_valid_losses

    def save(self, filepath: str):
        """
        Saves the model state and hyperparameters to the specified filepath.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hyperparameters': self.params
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Loads the model state from the specified filepath.
        
        Returns:
            hyperparameters (dict): The hyperparameters stored in the checkpoint.
        """
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return checkpoint.get("hyperparameters", {})