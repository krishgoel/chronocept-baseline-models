"""
Test script to verify the improved baseline implementation works correctly.
"""

import torch
import logging
from v2.models import SBERTFFNN, RoBERTaRegression, DeBERTaRegression
from v2.utils import ImprovedDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sbert_model():
    """Test SBERT model creation and forward pass."""
    logger.info("Testing SBERT model...")
    
    try:
        # Create model
        model = SBERTFFNN(
            sbert_model="all-MiniLM-L6-v2",
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            axis_encoding="single_sequence_markers",
            loss_type="skew_normal"
        )
        
        # Test forward pass with dummy data
        texts = ["This is a test sentence.", "Another test sentence."]
        axes_data = [
            {"main_outcome_axis": "test axis 1", "static_axis": "test axis 2"},
            {"main_outcome_axis": "test axis 3", "static_axis": "test axis 4"}
        ]
        
        with torch.no_grad():
            predictions = model(texts, axes_data)
        
        logger.info(f"SBERT model test passed. Output shape: {predictions.shape}")
        return True
        
    except Exception as e:
        logger.error(f"SBERT model test failed: {e}")
        return False

def test_roberta_model():
    """Test RoBERTa model creation and forward pass."""
    logger.info("Testing RoBERTa model...")
    
    try:
        # Create model
        model = RoBERTaRegression(
            model_name="roberta-base",
            pooling_type="mean",
            head_type="linear",
            axis_encoding="single_sequence_markers",
            loss_type="skew_normal",
            dropout=0.1
        )
        
        # Test forward pass with dummy data
        texts = ["This is a test sentence.", "Another test sentence."]
        axes_data = [
            {"main_outcome_axis": "test axis 1", "static_axis": "test axis 2"},
            {"main_outcome_axis": "test axis 3", "static_axis": "test axis 4"}
        ]
        
        with torch.no_grad():
            predictions = model(texts, axes_data)
        
        logger.info(f"RoBERTa model test passed. Output shape: {predictions.shape}")
        return True
        
    except Exception as e:
        logger.error(f"RoBERTa model test failed: {e}")
        return False

def test_data_loader():
    """Test data loader creation."""
    logger.info("Testing data loader...")
    
    try:
        # Create data loader
        data_loader = ImprovedDataLoader(
            benchmark="benchmark_1",
            axis_encoding="single_sequence_markers",
            batch_size=4,
            shuffle=False
        )
        
        train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
        
        # Test loading a batch
        batch = next(iter(train_loader))
        
        logger.info(f"Data loader test passed. Batch keys: {batch.keys()}")
        logger.info(f"Texts: {len(batch['texts'])}")
        logger.info(f"Targets shape: {batch['targets'].shape}")
        return True
        
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")
        return False

def test_loss_functions():
    """Test loss functions."""
    logger.info("Testing loss functions...")
    
    try:
        from v2.models.losses import SkewNormalNLL, GaussianNLL, MSELoss
        
        # Create dummy predictions and targets
        predictions = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        
        # Test skew-normal NLL
        skew_loss = SkewNormalNLL()
        loss_value = skew_loss(predictions, targets)
        logger.info(f"Skew-normal NLL test passed. Loss: {loss_value.item():.4f}")
        
        # Test Gaussian NLL
        gauss_loss = GaussianNLL()
        gauss_predictions = predictions[:, :2]  # Only mu, sigma
        gauss_targets = targets[:, :2]
        loss_value = gauss_loss(gauss_predictions, gauss_targets)
        logger.info(f"Gaussian NLL test passed. Loss: {loss_value.item():.4f}")
        
        # Test MSE
        mse_loss = MSELoss()
        loss_value = mse_loss(predictions, targets)
        logger.info(f"MSE test passed. Loss: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Loss functions test failed: {e}")
        return False

def test_metrics():
    """Test metrics computation."""
    logger.info("Testing metrics...")
    
    try:
        from v2.utils.metrics import evaluate_model_comprehensive
        
        # Create dummy predictions and targets
        predictions = torch.randn(10, 3).numpy()
        targets = torch.randn(10, 3).numpy()
        
        # Test comprehensive evaluation
        metrics = evaluate_model_comprehensive(predictions, targets, "skew_normal")
        
        logger.info(f"Metrics test passed. Available metrics: {list(metrics.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting implementation tests...")
    
    tests = [
        test_sbert_model,
        test_roberta_model,
        test_data_loader,
        test_loss_functions,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✅ All tests passed! Implementation is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
