import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from driftguard import DriftGuard, Config, DriftConfig, MonitorConfig
import json

# Load example dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def add_noise(data, noise_level=0.5):
    """Add random noise to simulate drift."""
    return pd.DataFrame(
        data.values + np.random.normal(0, noise_level, size=data.shape),
        columns=data.columns
    )

# Example monitoring usage
if __name__ == "__main__":
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
    print("\nMonitoring normal data...")
    normal_results = guard.monitor(X_test, y_test)
    print("Normal data results:")
    print(f"Drift detected: {normal_results['drift_detected']}")
    print(f"Performance metrics: {normal_results['performance']}")
    
    # Monitor drifted data
    print("\nMonitoring drifted data...")
    X_drift = add_noise(X_test, noise_level=0.5)
    drift_results = guard.monitor(X_drift, y_test)
    print("Drift data results:")
    print(f"Drift detected: {drift_results['drift_detected']}")
    print(f"Performance metrics: {drift_results['performance']}")
    
    # Get monitoring summary
    summary = guard.get_monitoring_summary()
    print("\nMonitoring Summary:")
    print(json.dumps(summary, indent=2))