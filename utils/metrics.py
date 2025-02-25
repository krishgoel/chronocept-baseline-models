import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from typing import Tuple

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
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
    
    return mse, mae, r2, nll, crps
