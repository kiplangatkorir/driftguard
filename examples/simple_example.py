import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftguard import DriftGuard, Config, DriftConfig, MonitorConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import logging
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelMonitoring')

def setup_monitoring_example():
    """Set up the monitoring example with the Iris dataset."""
    # Load environment variables
    load_dotenv()
    
    # Load and prepare data
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model and return it."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def simulate_drift(X_prod, drift_magnitude=0.5):
    """Simulate drift in production data."""
    drift = np.random.normal(0, drift_magnitude, size=X_prod.shape)
    return pd.DataFrame(
        X_prod.values + drift,
        columns=X_prod.columns
    )

def main():
    """Main function to run the monitoring example."""
    logger.info("Starting model monitoring example...")
    
    # Setup example data
    X_train, X_test, y_train, y_test = setup_monitoring_example()
    
    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    
    # Configure DriftGuard
    config = Config(
        drift=DriftConfig(
            methods=["ks", "anderson"],
            thresholds={"ks": 0.05, "anderson": 0.05}
        ),
        monitor=MonitorConfig(
            metrics=["accuracy", "f1", "roc_auc"],
            window_size=100
        )
    )
    
    # Initialize DriftGuard
    guard = DriftGuard(
        model=model,
        reference_data=X_train,
        config=config
    )
    
    # Monitor normal data
    logger.info("\nMonitoring normal data...")
    normal_results = guard.monitor(
        X_test,
        y_test
    )
    logger.info("Normal data results:")
    logger.info(f"Drift detected: {normal_results['drift_detected']}")
    logger.info(f"Performance metrics: {normal_results['performance']}")
    
    # Simulate and monitor drifted data
    logger.info("\nMonitoring drifted data...")
    X_drift = simulate_drift(X_test, drift_magnitude=0.5)
    drift_results = guard.monitor(
        X_drift,
        y_test
    )
    logger.info("Drift data results:")
    logger.info(f"Drift detected: {drift_results['drift_detected']}")
    logger.info(f"Performance metrics: {drift_results['performance']}")
    
    # Get monitoring summary
    summary = guard.get_monitoring_summary()
    logger.info("\nMonitoring Summary:")
    logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()