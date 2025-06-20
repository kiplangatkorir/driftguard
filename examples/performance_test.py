"""Performance test for optimized drift detection"""
import logging
import time
import traceback
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting performance test...")
        
        # Generate test data
        logger.info("Generating test data...")
        np.random.seed(42)
        
        # Generate synthetic data with some drift
        ref_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 10000) 
            for i in range(10)  # Reduced from 50 to 10 for faster testing
        })
        
        # Introduce some drift in test data
        test_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0.1, 1.1, 5000)  # Reduced size for testing
            for i in range(10)  # Reduced from 50 to 10 for faster testing
        })
        
        # Create synthetic labels
        y_train = (ref_data['feature_0'] > 0).astype(int)
        
        # Initialize and train a simple model
        logger.info("Training model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(ref_data, y_train)
        
        # Import after potential dependency issues are caught
        try:
            from driftguard.core.drift import DriftDetector
            from driftguard.core.config import DriftConfig
        except ImportError as e:
            logger.error(f"Failed to import DriftGuard modules: {e}")
            logger.error(traceback.format_exc())
            return
        
        logger.info("Initializing detector...")
        config = DriftConfig(methods=['ks', 'psi', 'jsd'])
        detector = DriftDetector(config)
        
        try:
            detector.initialize(ref_data)
            logger.info("Attaching model for SHAP calculations...")
            detector.attach_model(model)
            
            # Test performance
            logger.info("\nRunning optimized detection...")
            logger.info(f"Reference data: {ref_data.shape}")
            logger.info(f"Test data: {test_data.shape}")
            logger.info(f"Methods: {config.methods}")
            logger.info("Batch size: 1000")
            
            start = time.time()
            results = detector.detect(test_data, batch_size=1000)
            duration = time.time() - start
            
            # Show results
            logger.info("\n=== Results ===")
            logger.info(f"Total time: {duration:.2f} seconds")
            if results:
                drift_count = len([r for r in results if hasattr(r, 'score') and hasattr(r, 'threshold') and r.score > r.threshold])
                logger.info(f"Features with drift: {drift_count}/{len(results)}")
                
                # Sort by score if possible
                try:
                    sorted_results = sorted(
                        [r for r in results if hasattr(r, 'score')], 
                        key=lambda x: x.score, 
                        reverse=True
                    )
                    logger.info("\nTop 5 drifted features:")
                    for r in sorted_results[:5]:
                        if hasattr(r, 'feature') and hasattr(r, 'score') and hasattr(r, 'threshold'):
                            msg = f"- {r.feature}: {getattr(r, 'method', 'N/A')}={r.score:.3f} (threshold={r.threshold:.3f})"
                            if hasattr(r, 'importance_change') and r.importance_change is not None:
                                msg += f"\n  Importance change: {r.importance_change:.4f}"
                            logger.info(msg)
                except Exception as e:
                    logger.error(f"Error processing results: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("No results returned from detect()")
                
        except Exception as e:
            logger.error(f"Error during drift detection: {e}")
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
