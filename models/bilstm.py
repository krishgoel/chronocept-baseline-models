import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class BiLSTMModel(BaseModel, nn.Module):
    """
    BiLSTM Model that inherits from BaseModel.
    
    This model is designed to work with sequential embeddings.
    Hyperparameters (to be tuned via grid search):
      - hidden_dim: Hidden dimension for the LSTM.
      - bidirectional: Whether the LSTM is bidirectional.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, bidirectional: bool = True):
        params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bidirectional": bidirectional
        }
        super().__init__(params)
        nn.Module.__init__(self)
        self.input_dim = input_dim      # Per-token feature dimension (e.g. 768)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.build_model()
    
    def build_model(self):
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.out = nn.Linear(out_dim, 3)
        logger.info(f"BiLSTM built with input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, bidirectional={self.bidirectional}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects input x of shape [batch, seq_length, input_dim].
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, (h, c) = self.lstm(x)
        if self.bidirectional:
            h_forward = h[-2, :, :]
            h_backward = h[-1, :, :]
            h_cat = torch.cat((h_forward, h_backward), dim=1)
        else:
            h_cat = h[-1, :, :]
        return self.out(h_cat)
    
    def fit(self, train_data, valid_data, lr: float, epochs: int = 30, device: torch.device = torch.device("cpu")):
        """
        Contains the training loop for the BiLSTM.
        
        Args:
            train_data: Tuple (X_train, y_train) as NumPy arrays. For sequential embeddings, 
                        X_train should have shape [N, seq_length, D].
            valid_data: Tuple (X_valid, y_valid) similarly.
            lr: Learning rate.
            epochs: Number of epochs.
            device: Device for training.
        
        Returns:
            Tuple (train_losses, valid_losses).
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_losses = []
        valid_losses = []
        best_val_loss = float("inf")
        best_model_state = None
        
        for epoch in range(epochs):
            nn.Module.train.__get__(self)(True)
            optimizer.zero_grad()
            X_train, y_train = train_data
            inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
            targets = torch.tensor(y_train, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            nn.Module.train.__get__(self)(False)
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
            logger.info("Loaded best BiLSTM model state based on validation loss.")
        return train_losses, valid_losses
    
    def train(self, *args, **kwargs):
        """
        Overloaded train method.
        
        - If called with no arguments or a single boolean (e.g., self.train(False)),
          it toggles the training mode by directly invoking nn.Module.train.
        - If called with training data, it calls fit() and returns the losses.
        """
        if (not args and not kwargs) or (len(args) == 1 and isinstance(args[0], bool)):
            return nn.Module.train(self, *args, **kwargs)
        return self.fit(*args, **kwargs)
    
    def evaluate(self, test_data, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Evaluates the BiLSTM model.
        
        Args:
            test_data: Tuple (X_test, y_test) as NumPy arrays.
                      For sequential embeddings, X_test should have shape [N, seq_length, D].
            device: Device for evaluation.
        
        Returns:
            Predictions as a NumPy array.
        """
        self.to(device)
        nn.Module.train.__get__(self)(False)
        X_test, _ = test_data
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
        return outputs.cpu().numpy()
    
    def grid_search(self, train_data, valid_data, grid_params: dict, epochs: int, device: torch.device):
        """
        Performs grid search over hyperparameters.
        
        Expected keys in grid_params:
            - 'hidden_dim': list of ints
            - 'lr': list of floats
        
        Returns:
            Tuple (best_model, best_params, best_train_losses, best_valid_losses).
        """
        best_val_loss = float("inf")
        best_model = None
        best_params = {}
        best_train_losses = []
        best_valid_losses = []
        
        for hd in grid_params.get("hidden_dim", []):
            for lr in grid_params.get("lr", []):
                logger.info(f"BiLSTM Grid search: hidden_dim={hd}, lr={lr}")
                model = BiLSTMModel(self.input_dim, hidden_dim=hd, bidirectional=self.bidirectional)
                train_losses, valid_losses = model.train(train_data, valid_data, lr=lr, epochs=epochs, device=device)
                final_val_loss = valid_losses[-1]
                logger.info(f"Final validation loss: {final_val_loss:.4f} for hidden_dim={hd}, lr={lr}")
                
                plt.figure()
                plt.plot(valid_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"BiLSTM Config: hd={hd}, lr={lr}")
                plt.legend()
                plt.show()
                
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_model = model
                    best_params = {"hidden_dim": hd, "lr": lr}
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
        logger.info(f"BiLSTM model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Loads the model state from the specified filepath.
        
        Returns:
            The hyperparameters stored in the checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"BiLSTM model loaded from {filepath}")
        return checkpoint.get("hyperparameters", {})