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
            f'feature_{i}': np.random.normal(0, 1, 100) 
            for i in range(2)  # Reduced to 2 features with 100 samples
        })
        logger.debug(f"Reference data shape: {ref_data.shape}")
        logger.debug(f"Reference data head:\n{ref_data.head()}")
        
        # Introduce some drift in test data
        test_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0.1, 1.1, 50)  # Reduced to 50 samples
            for i in range(2)  # Reduced to 2 features with 50 samples
        })
        logger.debug(f"Test data shape: {test_data.shape}")
        logger.debug(f"Test data head:\n{test_data.head()}")
        
        # Create synthetic labels
        y_train = (ref_data['feature_0'] > 0).astype(int)
        logger.debug(f"Labels shape: {y_train.shape}")
        logger.debug(f"Labels value counts:\n{y_train.value_counts()}")
        
        # Initialize and train a simple model
        logger.info("Training model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(ref_data, y_train)
        logger.debug(f"Model trained. Feature importances:\n{model.feature_importances_}")
        
        # Import after potential dependency issues are caught
        try:
            from driftguard.core.drift import DriftDetector
            from driftguard.core.config import DriftConfig
            logger.debug("Successfully imported DriftGuard modules")
        except ImportError as e:
            logger.error(f"Failed to import DriftGuard modules: {e}")
            logger.error(traceback.format_exc())
            return
        
        logger.info("Initializing detector...")
        config = DriftConfig(methods=['ks', 'psi', 'jsd'])
        detector = DriftDetector(config)
        logger.debug(f"Detector initialized with methods: {config.methods}")
        
        try:
            detector.initialize(ref_data)
            logger.debug("Detector initialized successfully")
            logger.debug(f"Feature types: {detector.feature_types}")
            logger.debug(f"Reference stats: {detector.reference_stats}")
            logger.info("Attaching model for SHAP calculations...")
            detector.attach_model(model)
            logger.debug("Model attached successfully")
            
            # Test performance
            logger.info("\nRunning optimized detection...")
            logger.info(f"Reference data: {ref_data.shape}")
            logger.info(f"Test data: {test_data.shape}")
            logger.info(f"Methods: {config.methods}")
            logger.info("Batch size: 1000")
            
            start = time.time()
            logger.debug("Starting detection with batch size 1000")
            results = detector.detect(test_data, batch_size=1000)
            duration = time.time() - start
            logger.debug(f"Detection completed in {duration:.2f} seconds")
            
            # Show results
            logger.info("\n=== Results ===")
            logger.info(f"Total time: {duration:.2f} seconds")
            if results:
                logger.debug(f"Raw results: {results}")
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
