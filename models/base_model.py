import abc
import logging

class BaseModel(abc.ABC):
    def __init__(self, params: dict):
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    @abc.abstractmethod
    def build_model(self):
        """Construct the model architecture."""
        pass

    @abc.abstractmethod
    def train(self, train_data, valid_data):
        """Train the model on train_data and validate on valid_data.
        
        Should include logging of loss per epoch and any other relevant metrics.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model on test_data and compute metrics such as MSE, MAE, R2, NLL, and CRPS."""
        pass

    @abc.abstractmethod
    def grid_search(self, train_data, valid_data, param_grid: dict):
        """Perform a grid search over hyperparameters defined in param_grid."""
        pass

    # def save(self, filepath: str):
    #     """Save the model state."""
    #     # Implementation can be model-specific
    #     pass

    # def load(self, filepath: str):
    #     """Load the model state."""
    #     # Implementation can be model-specific
    #     pass
