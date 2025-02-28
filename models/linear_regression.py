import logging
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from models.base_model import BaseModel
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class LinearRegressionModel(BaseModel):
    """
    Linear Regression baseline that adheres to the BaseModel interface.
    This model uses scikit-learn's MultiOutputRegressor to predict three target values.
    
    Hyperparameters (tunable via grid search):
      - estimator__fit_intercept: Whether to calculate the intercept for this model.
      
    Note: Because MultiOutputRegressor wraps a base estimator, hyperparameters for grid search must be prefixed with "estimator__".
    """
    def __init__(self, params: Dict = None):
        if params is None:
            params = {
                "estimator__fit_intercept": True
            }
        super().__init__(params)
        self.params = params
        self.model = None

    def build_model(self):
        """Constructs the base model."""
        base_lr = LinearRegression()
        self.model = MultiOutputRegressor(base_lr)
        logger.info("Linear Regression model built (wrapped in MultiOutputRegressor).")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Fits the Linear Regression model using the provided training data. This method bypasses grid search.
        
        Args:
            X_train: Feature matrix (NumPy array).
            y_train: Target values (NumPy array).
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        logger.info("Linear Regression model trained on provided data.")

    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, param_grid: Dict = None, cv: int = 5) -> Tuple['LinearRegressionModel', Dict]:
        """
        Performs grid search using GridSearchCV to tune hyperparameters.
        
        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_valid: Validation feature matrix.
            y_valid: Validation targets.
            param_grid: Dictionary of hyperparameters for grid search.
                        (Default: {'estimator__fit_intercept': [True, False]})
            cv: Number of cross-validation folds.
        
        Returns:
            A tuple (best_model, best_params) where best_model is an instance of LinearRegressionModel
            with the best found hyperparameters and best_params is a dict.
        """
        if param_grid is None:
            param_grid = {
                'estimator__fit_intercept': [True, False]
            }
        
        if self.model is None:
            self.build_model()
        
        logger.info("Starting grid search for Linear Regression...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info("Grid search complete.")
        logger.info(f"Best Hyperparameters: {best_params}")
        
        y_valid_pred = best_estimator.predict(X_valid)
        val_mse = mean_squared_error(y_valid, y_valid_pred)
        logger.info(f"Validation MSE for best model: {val_mse:.4f}")
        
        best_model = LinearRegressionModel(params=best_params)
        best_model.model = best_estimator
        return best_model, best_params

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Evaluates the model on the test data.
        
        Args:
            X_test: Test feature matrix.
            y_test: True test target values.
        
        Returns:
            A tuple containing (MSE, MAE, R2, NLL, CRPS).
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        residuals = y_test - y_pred
        std_dev = np.std(residuals)
        if std_dev == 0:
            std_dev = 1e-6
        nll = -np.mean(norm.logpdf(residuals, scale=std_dev))
        crps = np.mean(np.abs(residuals))
        logger.info("Evaluation Metrics:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R2: {r2:.4f}")
        logger.info(f"NLL: {nll:.4f}")
        logger.info(f"CRPS: {crps:.4f}")
        return mse, mae, r2, nll, crps

    def save(self, filepath: str):
        """
        Saves the trained model and hyperparameters to the specified filepath.
        """
        checkpoint = {
            'model': self.model,
            'hyperparameters': self.params
        }
        joblib.dump(checkpoint, filepath)
        logger.info(f"Linear Regression model saved to {filepath}")

    def load(self, filepath: str) -> Dict:
        """
        Loads the model state and hyperparameters from the specified filepath.
        
        Returns:
            The hyperparameters loaded from the checkpoint.
        """
        checkpoint = joblib.load(filepath)
        self.model = checkpoint['model']
        self.params = checkpoint.get('hyperparameters', {})
        logger.info(f"Linear Regression model loaded from {filepath}")
        return self.params