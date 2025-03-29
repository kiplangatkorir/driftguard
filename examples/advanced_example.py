import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftguard import DriftGuard, Config, DriftConfig, MonitorConfig, AlertConfig, StateManager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import logging
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AdvancedMonitoring')

def setup_monitoring_example():
    """Set up the monitoring example with the Wine dataset."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_initial_model(X_train, y_train):
    """Train initial model version."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_improved_model(X_train, y_train):
    """Train improved model version with different parameters."""
    model = LogisticRegression(
        C=0.1,
        max_iter=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def simulate_data_drift(X, magnitude=0.5):
    """Simulate data drift by adding noise to features."""
    drift = np.random.normal(0, magnitude, size=X.shape)
    return pd.DataFrame(
        X.values + drift,
        columns=X.columns
    )

def main():
    """Main function to run the advanced monitoring example."""
    # Setup
    X_train, X_test, y_train, y_test = setup_monitoring_example()
    
    # Initialize state manager for version control
    state_manager = StateManager()
    
    # Train initial model
    logger.info("Training initial model...")
    model_v1 = train_initial_model(X_train, y_train)
    
    # Configure DriftGuard for v1
    config_v1 = Config(
        drift=DriftConfig(
            methods=["ks", "anderson", "wasserstein"],
            thresholds={"ks": 0.05, "anderson": 0.05, "wasserstein": 0.1},
            feature_importance=True
        ),
        monitor=MonitorConfig(
            metrics=["accuracy", "f1", "roc_auc", "precision", "recall"],
            window_size=100,
            performance_threshold=0.1
        ),
        alerts=AlertConfig(
            email="kiplangatgilbert00@gmail.com",
            alert_threshold=0.5,
            rate_limit=300  # 5 minutes
        )
    )
    
    # Initialize DriftGuard v1
    guard_v1 = DriftGuard(
        model=model_v1,
        reference_data=X_train,
        config=config_v1
    )
    
    # Save initial version
    state_manager.save_version(guard_v1, "v1")
    
    # Monitor with v1
    logger.info("\nMonitoring with model v1...")
    normal_results_v1 = guard_v1.monitor(X_test, y_test)
    logger.info("Normal data results (v1):")
    logger.info(f"Drift detected: {normal_results_v1['drift_detected']}")
    logger.info(f"Performance metrics: {normal_results_v1['performance']}")
    
    # Simulate drift
    logger.info("\nSimulating drift...")
    X_drift = simulate_data_drift(X_test, magnitude=0.5)
    drift_results_v1 = guard_v1.monitor(X_drift, y_test)
    logger.info("Drift data results (v1):")
    logger.info(f"Drift detected: {drift_results_v1['drift_detected']}")
    logger.info(f"Performance metrics: {drift_results_v1['performance']}")
    
    # Train improved model
    logger.info("\nTraining improved model...")
    model_v2 = train_improved_model(X_train, y_train)
    
    # Configure DriftGuard for v2 with stricter thresholds
    config_v2 = Config(
        drift=DriftConfig(
            methods=["ks", "anderson", "wasserstein"],
            thresholds={"ks": 0.01, "anderson": 0.01, "wasserstein": 0.05},
            feature_importance=True
        ),
        monitor=MonitorConfig(
            metrics=["accuracy", "f1", "roc_auc", "precision", "recall"],
            window_size=50,
            performance_threshold=0.05
        ),
        alerts=AlertConfig(
            email="kiplangatgilbert00@gmail.com",
            alert_threshold=0.3,
            rate_limit=300
        )
    )
    
    # Initialize DriftGuard v2
    guard_v2 = DriftGuard(
        model=model_v2,
        reference_data=X_train,
        config=config_v2
    )
    
    # Save new version
    state_manager.save_version(guard_v2, "v2")
    
    # Monitor with v2
    logger.info("\nMonitoring with model v2...")
    normal_results_v2 = guard_v2.monitor(X_test, y_test)
    logger.info("Normal data results (v2):")
    logger.info(f"Drift detected: {normal_results_v2['drift_detected']}")
    logger.info(f"Performance metrics: {normal_results_v2['performance']}")
    
    # Monitor drift with v2
    drift_results_v2 = guard_v2.monitor(X_drift, y_test)
    logger.info("Drift data results (v2):")
    logger.info(f"Drift detected: {drift_results_v2['drift_detected']}")
    logger.info(f"Performance metrics: {drift_results_v2['performance']}")
    
    # Compare versions
    logger.info("\nComparing versions...")
    v1_summary = guard_v1.get_monitoring_summary()
    v2_summary = guard_v2.get_monitoring_summary()
    
    logger.info("\nVersion 1 Summary:")
    logger.info(json.dumps(v1_summary, indent=2))
    logger.info("\nVersion 2 Summary:")
    logger.info(json.dumps(v2_summary, indent=2))
    
    # List all versions
    versions = state_manager.list_versions()
    logger.info("\nAll versions:")
    for version in versions:
        logger.info(f"Version: {version['name']}")
        logger.info(f"Created: {version['timestamp']}")
        logger.info(f"Config: {version['config']}\n")

if __name__ == "__main__":
    main()