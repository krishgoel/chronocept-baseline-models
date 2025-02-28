import logging
import numpy as np
import joblib
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from models.base_model import BaseModel
from typing import Dict, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XGBoostRegressionModel(BaseModel):
    """
    XGBoost Regression Model that adheres to the BaseModel interface.
    
    This model uses scikit-learn's XGBoost regressor (wrapped in a MultiOutputRegressor) to predict three target values.
    
    Hyperparameters must be supplied via the params dictionary.
    Example:
        params = {
            'estimator__n_estimators': 100,
            'estimator__max_depth': 3,
            'estimator__learning_rate': 0.1
        }
    """
    def __init__(self, params: Dict):
        """
        Initialize the XGBoostRegressionModel with given hyperparameters.
        
        Args:
            params (dict): Dictionary containing hyperparameters.
        """
        super().__init__(params)
        self.params = params
        self.model = None

    def build_model(self):
        """
        Constructs the XGBoost model wrapped in a MultiOutputRegressor using the provided hyperparameters.
        """
        base_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.params["estimator__n_estimators"],
            max_depth=self.params["estimator__max_depth"],
            learning_rate=self.params["estimator__learning_rate"]
        )
        self.model = MultiOutputRegressor(base_xgb)
        logger.info("XGBoost model built (wrapped in MultiOutputRegressor).")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Trains the XGBoost model using the provided training data.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training targets.
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        logger.info("XGBoost model trained on provided data.")

    def grid_search(self, 
                    X_train: np.ndarray, y_train: np.ndarray, 
                    X_valid: np.ndarray, y_valid: np.ndarray, 
                    param_grid: Dict, cv: int = 5) -> Tuple['XGBoostRegressionModel', Dict]:
        """
        Performs grid search using GridSearchCV to tune hyperparameters.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training targets.
            X_valid (np.ndarray): Validation feature matrix.
            y_valid (np.ndarray): Validation targets.
            param_grid (dict): Dictionary of hyperparameters for grid search.
                               Example: {'estimator__n_estimators': [50, 100],
                                         'estimator__max_depth': [3, 5],
                                         'estimator__learning_rate': [0.1, 0.01]}
            cv (int): Number of cross-validation folds.
        
        Returns:
            A tuple (best_model, best_params), where best_model is an instance of XGBoostRegressionModel with the best found hyperparameters and best_params is a dict.
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Starting grid search for XGBoost...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=4,
            n_jobs=10
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info("Grid search complete.")
        logger.info(f"Best Hyperparameters: {best_params}")
        
        y_valid_pred = best_estimator.predict(X_valid)
        val_mse = mean_squared_error(y_valid, y_valid_pred)
        logger.info(f"Validation MSE for best XGBoost model: {val_mse:.4f}")
        
        best_model = XGBoostRegressionModel(params=best_params)
        best_model.model = best_estimator
        return best_model, best_params

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Evaluates the XGBoost model on test data.
        
        Args:
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): True test targets.
        
        Returns:
            A tuple (MSE, MAE, R2, NLL, CRPS).
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
        Saves the model state and hyperparameters to the specified filepath.
        
        Args:
            filepath (str): The path to save the model.
        """
        checkpoint = {
            'model': self.model,
            'hyperparameters': self.params
        }
        joblib.dump(checkpoint, filepath)
        logger.info(f"XGBoost model saved to {filepath}")

    def load(self, filepath: str) -> Dict:
        """
        Loads the model state and hyperparameters from the specified filepath.
        
        Args:
            filepath (str): The path from which to load the model.
        
        Returns:
            A dictionary of hyperparameters loaded from the checkpoint.
        """
        checkpoint = joblib.load(filepath)
        self.model = checkpoint['model']
        self.params = checkpoint.get('hyperparameters', {})
        logger.info(f"XGBoost model loaded from {filepath}")
        return self.params
