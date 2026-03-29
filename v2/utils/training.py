"""
Training utilities and experiment management.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages training process with logging, checkpointing, and early stopping."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 valid_loader,
                 test_loader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: torch.device,
                 save_dir: str = "checkpoints",
                 experiment_name: str = None):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment name
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.training_history = {
            'train_losses': [],
            'valid_losses': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.patience = 10
        self.patience_counter = 0
        
        logger.info(f"Training manager initialized for experiment: {experiment_name}")
        logger.info(f"Save directory: {self.experiment_dir}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            texts = batch['texts']
            axes_data = batch.get('axes_data', None)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            predictions = self.model(texts, axes_data)
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for batch in self.valid_loader:
                texts = batch['texts']
                axes_data = batch.get('axes_data', None)
                targets = batch['targets'].to(self.device)
                
                predictions = self.model(texts, axes_data)
                loss = self.loss_fn(predictions, targets)
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def train(self, epochs: int, warm_start_epochs: int = 0) -> Dict[str, Any]:
        """Train the model for specified epochs."""
        logger.info(f"Starting training for {epochs} epochs")
        if warm_start_epochs > 0:
            logger.info(f"Warm start training for {warm_start_epochs} epochs")
        
        # Warm start training
        if warm_start_epochs > 0:
            warm_history = self._warm_start_training(warm_start_epochs)
        
        # Full training
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['valid_losses'].append(val_loss)
            self.training_history['learning_rates'].append(current_lr)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Log progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        # Create training plots
        self._create_training_plots()
        
        # Prepare return history
        history = self.training_history.copy()
        if warm_start_epochs > 0:
            history['warm_start'] = warm_history
        
        return history
    
    def _warm_start_training(self, epochs: int) -> Dict[str, Any]:
        """Warm start training (xi only)."""
        # Temporarily replace loss with xi-only MSE to avoid shape issues
        original_loss_fn = self.loss_fn

        class XiOnlyLoss(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # If prediction has 6-dim param_gauss head, use mu_xi at index 0
                if pred.dim() >= 2 and pred.size(-1) >= 1:
                    if pred.size(-1) == 6:
                        xi_pred = pred[:, 0]
                    else:
                        xi_pred = pred[:, 0]
                else:
                    xi_pred = pred.squeeze(-1)
                xi_true = target[:, 0]
                return torch.nn.functional.mse_loss(xi_pred, xi_true)

        self.loss_fn = XiOnlyLoss()
        
        warm_history = {'train_losses': [], 'valid_losses': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            warm_history['train_losses'].append(train_loss)
            warm_history['valid_losses'].append(val_loss)
            
            logger.info(f"Warm start epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Restore original loss function
        self.loss_fn = original_loss_fn
        
        return warm_history
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if is_best:
            checkpoint_path = self.experiment_dir / "best_model.pt"
        elif is_final:
            checkpoint_path = self.experiment_dir / "final_model.pt"
        else:
            checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{self.epoch}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _create_training_plots(self):
        """Create training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_losses'], label='Train Loss')
        axes[0, 0].plot(self.training_history['valid_losses'], label='Valid Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rates'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Loss difference
        if len(self.training_history['train_losses']) > 1:
            loss_diff = np.array(self.training_history['valid_losses']) - np.array(self.training_history['train_losses'])
            axes[1, 0].plot(loss_diff)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Validation - Training Loss')
            axes[1, 0].set_title('Overfitting Indicator')
            axes[1, 0].grid(True)
        
        # Best validation loss
        best_epoch = np.argmin(self.training_history['valid_losses'])
        axes[1, 1].axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch}')
        axes[1, 1].plot(self.training_history['valid_losses'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Validation Loss with Best Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save as both PNG and PDF
        plt.savefig(self.experiment_dir / "training_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.experiment_dir / "training_plots.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {self.experiment_dir / 'training_plots.png'} and .pdf")


class ExperimentLogger:
    """Logs experiment results and metadata."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.results = {}
    
    @staticmethod
    def _to_serializable(obj):
        """Recursively convert numpy/torch types to Python natives for JSON."""
        try:
            import numpy as np  # local import to avoid hard dep at module load
            import torch  # type: ignore
        except Exception:
            np = None  # type: ignore
            torch = None  # type: ignore
        
        if obj is None:
            return None
        
        # Basic scalars
        if isinstance(obj, (bool, int, float, str)):
            return obj
        
        # Numpy scalars/arrays
        if np is not None:
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        
        # Torch tensors
        if torch is not None:
            if isinstance(obj, torch.Tensor):
                return ExperimentLogger._to_serializable(obj.detach().cpu().numpy())
        
        # Mappings
        if isinstance(obj, dict):
            return {str(k): ExperimentLogger._to_serializable(v) for k, v in obj.items()}
        
        # Sequences
        if isinstance(obj, (list, tuple)):
            return [ExperimentLogger._to_serializable(v) for v in obj]
        
        # Fallback to string
        try:
            return float(obj)  # try numeric cast
        except Exception:
            return str(obj)
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.results['config'] = self._to_serializable(config)
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.results['config'], f, indent=2)
        logger.info(f"Experiment config saved to {config_path}")
    
    def log_training_results(self, training_history: Dict[str, Any]):
        """Log training results."""
        self.results['training'] = self._to_serializable(training_history)
        training_path = self.experiment_dir / "training_results.json"
        with open(training_path, 'w') as f:
            json.dump(self.results['training'], f, indent=2)
        logger.info(f"Training results saved to {training_path}")
    
    def log_evaluation_results(self, metrics: Dict[str, float]):
        """Log evaluation results."""
        self.results['evaluation'] = self._to_serializable(metrics)
        eval_path = self.experiment_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(self.results['evaluation'], f, indent=2)
        logger.info(f"Evaluation results saved to {eval_path}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.results['model'] = self._to_serializable(model_info)
        model_path = self.experiment_dir / "model_info.json"
        with open(model_path, 'w') as f:
            json.dump(self.results['model'], f, indent=2)
        logger.info(f"Model info saved to {model_path}")
    
    def create_summary_report(self):
        """Create a summary report of the experiment."""
        summary = {
            'experiment_name': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'results': self._to_serializable(self.results)
        }
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary
