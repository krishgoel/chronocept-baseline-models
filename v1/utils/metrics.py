import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, pearsonr, spearmanr
from typing import Tuple

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """
    Evaluates model performance by computing multiple regression metrics.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values from the model.
    
    Returns:
        Tuple containing:
            - MSE (float): Mean Squared Error.
            - MAE (float): Mean Absolute Error.
            - R2 (float): R^2 Score.
            - NLL (float): Negative Log Likelihood.
            - CRPS (float): Continuous Ranked Probability Score.
            - Pearson Correlation Coefficient (float): Average Pearson correlation across dimensions.
            - Spearman Correlation Coefficient (float): Average Spearman correlation across dimensions.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    residuals = y_true - y_pred
    std_dev = np.std(residuals)
    if std_dev == 0:
        std_dev = 1e-6  # Avoid division by zero
    
    nll = -np.mean(norm.logpdf(residuals, scale=std_dev))
    
    crps = np.mean(np.abs(residuals))
    
    pearson_corrs = []
    spearman_corrs = []
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        for i in range(y_true.shape[1]):
            p_corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            s_corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
            pearson_corrs.append(p_corr)
            spearman_corrs.append(s_corr)
        
        pearson_corr = np.mean(pearson_corrs)
        spearman_corr = np.mean(spearman_corrs)
    else:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
        spearman_corr, _ = spearmanr(y_true_flat, y_pred_flat)
    
    return mse, mae, r2, nll, crps, pearson_corr, spearman_corr
