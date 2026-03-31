"""
Improved metrics for distributional regression evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.stats import skewnorm
import logging

logger = logging.getLogger(__name__)


def compute_skew_normal_nll(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute negative log-likelihood for skew-normal distribution.
    
    Args:
        predictions: [N, 3] array of [xi, omega, alpha] predictions
        targets: [N, 3] array of [xi_true, omega_true, alpha_true] targets
    
    Returns:
        NLL value
    """
    xi_pred, omega_pred, alpha_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    xi_true, omega_true, alpha_true = targets[:, 0], targets[:, 1], targets[:, 2]
    
    # Ensure positive scale parameter
    omega_pred = np.maximum(omega_pred, 1e-6)
    
    # Compute skew-normal log-likelihood
    nll_values = []
    for i in range(len(predictions)):
        try:
            # Create skew-normal distribution
            dist = skewnorm(alpha_pred[i], loc=xi_pred[i], scale=omega_pred[i])
            
            # Compute log-likelihood for true parameters
            # We use the true xi as the observation
            log_likelihood = dist.logpdf(xi_true[i])
            nll_values.append(-log_likelihood)
        except:
            # Fallback to normal distribution if skew-normal fails
            nll_values.append(0.5 * np.log(2 * np.pi * omega_pred[i]**2) + 
                            0.5 * ((xi_true[i] - xi_pred[i]) / omega_pred[i])**2)
    
    return np.mean(nll_values)


def compute_gaussian_nll(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute negative log-likelihood for Gaussian distribution.
    
    Args:
        predictions: [N, 2] array of [mu, sigma] predictions
        targets: [N, 2] array of [mu_true, sigma_true] targets
    
    Returns:
        NLL value
    """
    mu_pred, sigma_pred = predictions[:, 0], predictions[:, 1]
    mu_true, sigma_true = targets[:, 0], targets[:, 1]
    
    # Ensure positive sigma
    sigma_pred = np.maximum(sigma_pred, 1e-6)
    
    # Gaussian NLL
    nll = 0.5 * np.log(2 * np.pi * sigma_pred**2) + 0.5 * ((mu_true - mu_pred) / sigma_pred)**2
    return np.mean(nll)


def compute_crps(predictions: np.ndarray, targets: np.ndarray, 
                distribution_type: str = "skew_normal") -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).
    
    Args:
        predictions: [N, 3] array of distribution parameters
        targets: [N, 3] array of true parameters
        distribution_type: "skew_normal" or "gaussian"
    
    Returns:
        CRPS value
    """
    if distribution_type == "skew_normal":
        xi_pred, omega_pred, alpha_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        xi_true = targets[:, 0]
        
        # Ensure positive scale
        omega_pred = np.maximum(omega_pred, 1e-6)
        
        # Approximate CRPS using quantile regression
        quantiles = np.linspace(0.01, 0.99, 99)
        crps_values = []
        
        for i in range(len(predictions)):
            try:
                # Create skew-normal distribution
                dist = skewnorm(alpha_pred[i], loc=xi_pred[i], scale=omega_pred[i])
                
                # Compute quantiles
                q_values = dist.ppf(quantiles)
                
                # CRPS approximation
                crps = np.mean(np.abs(q_values - xi_true[i]))
                crps_values.append(crps)
            except:
                # Fallback to normal CRPS
                crps_values.append(omega_pred[i] * (1/np.sqrt(np.pi) - 2 * stats.norm.pdf((xi_true[i] - xi_pred[i]) / omega_pred[i])))
        
        return np.mean(crps_values)
    
    elif distribution_type == "gaussian":
        mu_pred, sigma_pred = predictions[:, 0], predictions[:, 1]
        mu_true = targets[:, 0]
        
        # Ensure positive sigma
        sigma_pred = np.maximum(sigma_pred, 1e-6)
        
        # Gaussian CRPS
        z = (mu_true - mu_pred) / sigma_pred
        crps = sigma_pred * (1/np.sqrt(np.pi) - 2 * stats.norm.pdf(z) - z * (2 * stats.norm.cdf(z) - 1))
        return np.mean(crps)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


def compute_per_param_rmse(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE for each parameter separately.
    
    Args:
        predictions: [N, 3] array of predictions
        targets: [N, 3] array of targets
    
    Returns:
        Dictionary with RMSE for each parameter
    """
    param_names = ['xi', 'omega', 'alpha']
    rmse_dict = {}
    
    for i, param_name in enumerate(param_names):
        rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
        rmse_dict[f'rmse_{param_name}'] = rmse
    
    return rmse_dict


def compute_spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute Spearman correlation for each parameter.
    
    Args:
        predictions: [N, 3] array of predictions
        targets: [N, 3] array of targets
    
    Returns:
        Dictionary with Spearman correlation for each parameter
    """
    param_names = ['xi', 'omega', 'alpha']
    spearman_dict = {}
    
    for i, param_name in enumerate(param_names):
        correlation, _ = stats.spearmanr(predictions[:, i], targets[:, i])
        spearman_dict[f'spearman_{param_name}'] = correlation
    
    return spearman_dict


def compute_pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute Pearson correlation for each parameter.
    
    Args:
        predictions: [N, 3] array of predictions
        targets: [N, 3] array of targets
    
    Returns:
        Dictionary with Pearson correlation for each parameter
    """
    param_names = ['xi', 'omega', 'alpha']
    pearson_dict = {}
    
    for i, param_name in enumerate(param_names):
        correlation, _ = stats.pearsonr(predictions[:, i], targets[:, i])
        pearson_dict[f'pearson_{param_name}'] = correlation
    
    return pearson_dict


def evaluate_model_comprehensive(predictions: np.ndarray, targets: np.ndarray, 
                               loss_type: str = "skew_normal") -> Dict[str, float]:
    """
    Comprehensive model evaluation with all relevant metrics.
    
    Args:
        predictions: [N, 3] array of predictions
        targets: [N, 3] array of targets
        loss_type: "skew_normal", "gaussian", or "mse"
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Basic regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metrics['r2'] = r2_score(targets, predictions)
    
    # Per-parameter RMSE
    rmse_dict = compute_per_param_rmse(predictions, targets)
    metrics.update(rmse_dict)
    
    # Correlation metrics
    spearman_dict = compute_spearman_correlation(predictions, targets)
    metrics.update(spearman_dict)
    
    pearson_dict = compute_pearson_correlation(predictions, targets)
    metrics.update(pearson_dict)
    
    # Distributional metrics
    if loss_type == "skew_normal":
        metrics['nll_skew'] = compute_skew_normal_nll(predictions, targets)
        metrics['crps_skew'] = compute_crps(predictions, targets, "skew_normal")
        
        # Also compute Gaussian baseline for comparison
        gaussian_pred = predictions[:, :2]  # Use only xi and omega
        gaussian_target = targets[:, :2]
        metrics['nll_gauss'] = compute_gaussian_nll(gaussian_pred, gaussian_target)
        metrics['crps_gauss'] = compute_crps(gaussian_pred, gaussian_target, "gaussian")
    
    elif loss_type == "gaussian":
        gaussian_pred = predictions[:, :2]  # Use only xi and omega
        gaussian_target = targets[:, :2]
        metrics['nll_gauss'] = compute_gaussian_nll(gaussian_pred, gaussian_target)
        metrics['crps_gauss'] = compute_crps(gaussian_pred, gaussian_target, "gaussian")
    
    else:  # MSE
        # For MSE, we can still compute some distributional metrics
        # Assume predictions are [xi, omega, alpha] and use xi as the main prediction
        xi_pred = predictions[:, 0]
        xi_true = targets[:, 0]
        
        # Simple Gaussian NLL assuming unit variance
        metrics['nll_simple'] = 0.5 * np.log(2 * np.pi) + 0.5 * np.mean((xi_true - xi_pred)**2)
        
        # Simple CRPS approximation
        metrics['crps_simple'] = np.mean(np.abs(xi_true - xi_pred))
    
    return metrics


def compute_parameter_statistics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed statistics for each parameter.
    
    Args:
        predictions: [N, 3] array of predictions
        targets: [N, 3] array of targets
    
    Returns:
        Dictionary with statistics for each parameter
    """
    param_names = ['xi', 'omega', 'alpha']
    stats_dict = {}
    
    for i, param_name in enumerate(param_names):
        pred_param = predictions[:, i]
        true_param = targets[:, i]
        
        stats_dict[param_name] = {
            'pred_mean': np.mean(pred_param),
            'pred_std': np.std(pred_param),
            'true_mean': np.mean(true_param),
            'true_std': np.std(true_param),
            'bias': np.mean(pred_param - true_param),
            'mae': np.mean(np.abs(pred_param - true_param)),
            'rmse': np.sqrt(np.mean((pred_param - true_param)**2)),
            'correlation': np.corrcoef(pred_param, true_param)[0, 1]
        }
    
    return stats_dict
