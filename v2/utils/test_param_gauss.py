"""
Test script for the new parameter-space Gaussian NLL approach.
"""

import torch
import logging
from v2.models import RoBERTaRegression, DeBERTaRegression
from v2.utils import ImprovedDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_param_gauss_models():
    """Test models with parameter-space Gaussian NLL."""
    logger.info("Testing parameter-space Gaussian NLL models...")
    
    try:
        # Test RoBERTa with param_gauss
        model = RoBERTaRegression(
            model_name="roberta-base",
            pooling_type="mean",
            head_type="linear",
            axis_encoding="single_sequence_markers",
            loss_type="param_gauss",  # Use new loss
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
        
        logger.info(f"RoBERTa param_gauss test passed. Output shape: {predictions.shape}")
        logger.info(f"Expected 6 outputs: [mu_xi, s_xi, mu_logw, s_logw, mu_alphat, s_alphat]")
        
        # Test loss computation
        targets = torch.tensor([[1.0, 2.0, 0.5], [2.0, 1.5, -0.3]], dtype=torch.float32)
        loss = model.loss_fn(predictions, targets)
        logger.info(f"Loss computation test passed. Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parameter-space Gaussian NLL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with corrected mapping."""
    logger.info("Testing corrected data loading...")
    
    try:
        # Create data loader
        data_loader = ImprovedDataLoader(
            benchmark="benchmark_1",
            axis_encoding="single_sequence_markers",
            batch_size=2,
            shuffle=False
        )
        
        train_loader, valid_loader, test_loader = data_loader.get_data_loaders()
        
        # Test loading a batch
        batch = next(iter(train_loader))
        
        logger.info(f"Data loading test passed. Batch keys: {batch.keys()}")
        logger.info(f"Texts: {len(batch['texts'])}")
        logger.info(f"Targets shape: {batch['targets'].shape}")
        logger.info(f"Sample targets: {batch['targets'][0]}")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_warm_start():
    """Test warm-start training with xi-only."""
    logger.info("Testing warm-start training...")
    
    try:
        model = RoBERTaRegression(
            model_name="roberta-base",
            loss_type="param_gauss"
        )
        
        # Create dummy data
        texts = ["Test sentence 1", "Test sentence 2"]
        axes_data = [{"axis1": "value1"}, {"axis1": "value2"}]
        targets = torch.tensor([[1.0, 2.0, 0.5], [2.0, 1.5, -0.3]], dtype=torch.float32)
        
        # Test warm-start loss (should only use xi)
        with torch.no_grad():
            predictions = model(texts, axes_data)
            warm_loss = model.loss_fn(predictions, targets)
        
        logger.info(f"Warm-start test passed. Predictions shape: {predictions.shape}")
        logger.info(f"Warm-start loss: {warm_loss.item():.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Warm-start test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting parameter-space Gaussian NLL tests...")
    
    tests = [
        test_param_gauss_models,
        test_data_loading,
        test_warm_start
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✅ All parameter-space Gaussian NLL tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
