"""
Loss functions for distributional regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SkewNormalNLL(nn.Module):
    """
    Negative Log-Likelihood loss for Skew-Normal distribution.
    
    Predicts xi (location), omega (scale), alpha (shape) parameters.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, 3] - [xi, omega, alpha]
            targets: [batch_size, 3] - [xi_true, omega_true, alpha_true]
        """
        xi_pred, omega_pred, alpha_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        xi_true, omega_true, alpha_true = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # Ensure positive scale parameter
        omega_pred = F.softplus(omega_pred) + self.eps
        
        # Compute skew-normal log-likelihood
        # Using the approximation: log-likelihood ≈ -0.5 * ((x - xi) / omega)^2 + log(2/omega) + log(phi(alpha * (x - xi) / omega))
        
        # Standardized residuals
        z = (xi_true - xi_pred) / omega_pred
        
        # Normal log-likelihood component
        normal_ll = -0.5 * z**2 - 0.5 * torch.log(2 * np.pi * omega_pred**2)
        
        # Skewness component (approximation)
        alpha_effect = alpha_pred * z
        skew_ll = torch.log(1 + torch.erf(alpha_effect / np.sqrt(2)) + self.eps)
        
        # Total log-likelihood
        log_likelihood = normal_ll + skew_ll
        
        # Return negative log-likelihood
        return -torch.mean(log_likelihood)


class GaussianNLL(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian distribution.
    
    Predicts mu (mean) and sigma (std) parameters.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, 2] - [mu, sigma]
            targets: [batch_size, 2] - [mu_true, sigma_true]
        """
        mu_pred, sigma_pred = predictions[:, 0], predictions[:, 1]
        mu_true, sigma_true = targets[:, 0], targets[:, 1]
        
        # Ensure positive sigma
        sigma_pred = F.softplus(sigma_pred) + self.eps
        
        # Gaussian NLL
        nll = 0.5 * torch.log(2 * np.pi * sigma_pred**2) + 0.5 * ((mu_true - mu_pred) / sigma_pred)**2
        
        return torch.mean(nll)


class MSELoss(nn.Module):
    """MSE loss for legacy baselines."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, targets)


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score loss.
    
    Approximates CRPS using quantile regression.
    """
    
    def __init__(self, n_quantiles: int = 10):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.quantiles = torch.linspace(0.1, 0.9, n_quantiles)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, 3] - [xi, omega, alpha] for skew-normal
            targets: [batch_size, 1] - true values
        """
        xi, omega, alpha = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        
        # Ensure positive scale
        omega = F.softplus(omega) + 1e-6
        
        # Compute quantiles of skew-normal distribution
        # This is a simplified approximation
        quantile_values = []
        for q in self.quantiles:
            # Approximate quantile using normal approximation with skewness correction
            z_q = torch.erfinv(2 * q - 1) * np.sqrt(2)
            skew_correction = alpha * (z_q**2 - 1) / 6
            quantile_val = xi + omega * (z_q + skew_correction)
            quantile_values.append(quantile_val)
        
        quantile_values = torch.stack(quantile_values, dim=1)  # [batch_size, n_quantiles]
        
        # CRPS approximation using quantile loss
        targets_expanded = targets.unsqueeze(1).expand(-1, self.n_quantiles)
        quantiles_expanded = self.quantiles.unsqueeze(0).expand(targets.size(0), -1).to(targets.device)
        
        # Quantile loss
        errors = targets_expanded - quantile_values
        loss = torch.mean(torch.max(
            quantiles_expanded * errors,
            (quantiles_expanded - 1) * errors
        ))
        
        return loss


class ParamSpaceGaussianNLL(nn.Module):
    """
    Per-parameter Gaussian NLL in parameter space with transformations.

    Predicts for each parameter a mean and log-std in a transformed space:
    - xi: identity transform (R)
    - omega: log-transform to enforce positivity (R)
    - alpha: tanh-bijection to a bounded range A, optimize in artanh(alpha/A) space (R)

    Expected predictions shape: [batch_size, 6]
      [mu_xi, s_xi, mu_logw, s_logw, mu_alphat, s_alphat]

    where sigma = softplus(s) + eps.
    """

    def __init__(self, alpha_bound: float = 5.0, eps: float = 1e-6):
        super().__init__()
        self.alpha_bound = float(alpha_bound)
        self.eps = float(eps)

    def _gaussian_nll(self, target: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(s) + self.eps
        z = (target - mu) / sigma
        # 0.5*log(2*pi) + log(sigma) + 0.5*z^2
        return 0.5 * torch.log(torch.tensor(2.0 * np.pi, device=mu.device, dtype=mu.dtype)) + torch.log(sigma) + 0.5 * (z ** 2)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 6] = [mu_xi, s_xi, mu_logw, s_logw, mu_alphat, s_alphat]
            targets:     [B, 3] = [xi_true, omega_true, alpha_true]
        """
        if predictions.shape[-1] != 6:
            raise ValueError(f"ParamSpaceGaussianNLL expects predictions with 6 dims, got {predictions.shape}")

        xi_true = targets[:, 0]
        omega_true = targets[:, 1]
        alpha_true = targets[:, 2]

        # Transforms
        logw_true = torch.log(torch.clamp(omega_true, min=self.eps))
        # Clip inside (-A, A) before inverse tanh to avoid numeric issues
        clipped_alpha = torch.clamp(alpha_true, min=-self.alpha_bound + self.eps, max=self.alpha_bound - self.eps)
        alphat_true = torch.atanh(clipped_alpha / self.alpha_bound)

        mu_xi, s_xi, mu_logw, s_logw, mu_alphat, s_alphat = (
            predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4], predictions[:, 5]
        )

        nll_xi = self._gaussian_nll(xi_true, mu_xi, s_xi)
        nll_logw = self._gaussian_nll(logw_true, mu_logw, s_logw)
        nll_alphat = self._gaussian_nll(alphat_true, mu_alphat, s_alphat)

        total_nll = nll_xi + nll_logw + nll_alphat
        return total_nll.mean()