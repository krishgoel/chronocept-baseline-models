"""
Improved data loader for the new baseline models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datasets import load_dataset
import logging
import random

logger = logging.getLogger(__name__)


class ChronoceptDataset(Dataset):
    """Dataset class for Chronocept data."""
    
    def __init__(self, data, axis_encoding: str = "single_sequence_markers"):
        self.data = data
        self.axis_encoding = axis_encoding
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract parent text
        parent_text = sample["parent_text"]
        
        # Extract axes data
        axes_data = sample.get("axes", {})
        
        # Extract targets - ensure the expected keys: xi, omega, alpha
        target_values = sample.get("target_values", {})
        # Some datasets may use alternate names; map them conservatively
        xi = target_values.get("xi", target_values.get("location", 0.0))
        omega = target_values.get("omega", target_values.get("scale", 1.0))
        alpha = target_values.get("alpha", target_values.get("skewness", 0.0))
        targets = np.array([xi, omega, alpha], dtype=np.float32)
        
        return {
            'texts': parent_text,
            'axes_data': axes_data,
            'targets': torch.tensor(targets, dtype=torch.float32)
        }


class ImprovedDataLoader:
    """
    Improved data loader that supports the new model architectures.
    """
    
    def __init__(self, 
                 benchmark: str = "benchmark_1",
                 axis_encoding: str = "single_sequence_markers",
                 max_length: int = 512,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 normalization: str = "zscore",
                 log_scale: float = 1.1):
        
        self.benchmark = benchmark
        self.axis_encoding = axis_encoding
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization = normalization
        self.log_scale = log_scale
        
        # Load dataset
        self.dataset = load_dataset("krishgoel/chronocept", benchmark)
        
        # Normalize targets
        self._normalize_targets()
        
        # Create datasets
        self.train_dataset = ChronoceptDataset(self.dataset["train"], axis_encoding)
        self.valid_dataset = ChronoceptDataset(self.dataset["validation"], axis_encoding)
        self.test_dataset = ChronoceptDataset(self.dataset["test"], axis_encoding)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Created data loaders for {benchmark} with {axis_encoding} axis encoding")
        logger.info(f"Train: {len(self.train_dataset)} samples")
        logger.info(f"Valid: {len(self.valid_dataset)} samples")
        logger.info(f"Test: {len(self.test_dataset)} samples")
    
    def _normalize_targets(self):
        """Normalize target values across all splits."""
        # Collect all targets
        all_targets = []
        for split in ["train", "validation", "test"]:
            for sample in self.dataset[split]:
                target_values = sample.get("target_values", {})
                targets = np.array([
                    target_values.get("xi", 0.0),
                    target_values.get("omega", 1.0),
                    target_values.get("alpha", 0.0)
                ])
                all_targets.append(targets)
        
        all_targets = np.array(all_targets)
        
        # Apply log transformation if specified
        if self.log_scale is not None:
            all_targets = np.log(all_targets + 1e-6) / np.log(self.log_scale)
        
        # Compute normalization statistics
        if self.normalization == "zscore":
            self.target_mean = np.mean(all_targets, axis=0)
            self.target_std = np.std(all_targets, axis=0)
            self.target_std = np.maximum(self.target_std, 1e-6)  # Avoid division by zero
        elif self.normalization == "minmax":
            self.target_min = np.min(all_targets, axis=0)
            self.target_max = np.max(all_targets, axis=0)
            self.target_range = self.target_max - self.target_min
            self.target_range = np.maximum(self.target_range, 1e-6)  # Avoid division by zero
        else:
            self.target_mean = np.zeros(3)
            self.target_std = np.ones(3)
        
        # Apply normalization to all splits
        for split in ["train", "validation", "test"]:
            for sample in self.dataset[split]:
                target_values = sample.get("target_values", {})
                targets = np.array([
                    target_values.get("xi", 0.0),
                    target_values.get("omega", 1.0),
                    target_values.get("alpha", 0.0)
                ])
                
                # Apply log transformation if specified
                if self.log_scale is not None:
                    targets = np.log(targets + 1e-6) / np.log(self.log_scale)
                
                # Apply normalization
                if self.normalization == "zscore":
                    targets = (targets - self.target_mean) / self.target_std
                elif self.normalization == "minmax":
                    targets = (targets - self.target_min) / self.target_range
                
                # Update the sample
                sample["target_values"] = {
                    "xi": targets[0],
                    "omega": targets[1],
                    "alpha": targets[2]
                }
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        texts = [item['texts'] for item in batch]
        axes_data = [item['axes_data'] for item in batch]
        targets = torch.stack([item['targets'] for item in batch])
        
        return {
            'texts': texts,
            'axes_data': axes_data,
            'targets': targets.float()  # Ensure float type
        }
    
    def denormalize_targets(self, normalized_targets: np.ndarray) -> np.ndarray:
        """Denormalize target values back to original scale."""
        targets = normalized_targets.copy()
        
        # Reverse normalization
        if self.normalization == "zscore":
            targets = targets * self.target_std + self.target_mean
        elif self.normalization == "minmax":
            targets = targets * self.target_range + self.target_min
        
        # Reverse log transformation if applied
        if self.log_scale is not None:
            targets = np.power(self.log_scale, targets) - 1e-6
        
        return targets
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        return self.train_loader, self.valid_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about the dataset."""
        return {
            'benchmark': self.benchmark,
            'axis_encoding': self.axis_encoding,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'normalization': self.normalization,
            'log_scale': self.log_scale,
            'train_samples': len(self.train_dataset),
            'valid_samples': len(self.valid_dataset),
            'test_samples': len(self.test_dataset),
            'target_mean': self.target_mean.tolist() if hasattr(self, 'target_mean') else None,
            'target_std': self.target_std.tolist() if hasattr(self, 'target_std') else None
        }


def create_data_loader(benchmark: str = "benchmark_1", 
                      axis_encoding: str = "single_sequence_markers",
                      **kwargs) -> ImprovedDataLoader:
    """
    Factory function to create a data loader.
    
    Args:
        benchmark: "benchmark_1" or "benchmark_2"
        axis_encoding: "no_axes", "single_sequence_markers", "sbert_concat", "9_pass_concat"
        **kwargs: Additional arguments for ImprovedDataLoader
    
    Returns:
        ImprovedDataLoader instance
    """
    return ImprovedDataLoader(
        benchmark=benchmark,
        axis_encoding=axis_encoding,
        **kwargs
    )
