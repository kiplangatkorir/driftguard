"""
Advanced example demonstrating DriftGuard's enhanced features including multivariate drift detection
and performance optimization.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from driftguard import Wrapper

# Generate synthetic data
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Convert to DataFrame with meaningful feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize DriftGuard with advanced settings
monitor = Wrapper(
    model=model,
    reference_data=X_train,
    alert_email="your.email@company.com",
    config={
        'drift_methods': ['ks', 'wasserstein', 'adwin', 'multivariate'],
        'batch_size': 1000,  # Enable batch processing
        'cache_size': 128,   # Set cache size for optimized performance
        'thresholds': {
            'ks': 0.05,
            'wasserstein': 0.1,
            'adwin': 0.05,
            'multivariate': 0.05
        }
    }
)

# Simulate drift by introducing gradual changes
def generate_drift_data(X, drift_magnitude=0.5):
    drift_X = X.copy()
    # Add gradual drift to some features
    for i in range(5):
        drift_X[f'feature_{i}'] += np.random.normal(
            loc=drift_magnitude,
            scale=0.1,
            size=len(X)
        )
    return drift_X

# Monitor data in batches
print("Monitoring original test data...")
results_original = monitor.monitor(X_test, y_test)
print("Original data drift results:", results_original)

# Generate and monitor drift data
print("\nMonitoring data with drift...")
X_drift = generate_drift_data(X_test)
results_drift = monitor.monitor(X_drift, y_test)
print("Drift data results:", results_drift)

# Example of accessing detailed drift statistics
print("\nDetailed drift analysis:")
for report in results_drift.drift_reports:
    print(f"\nMethod: {report.method}")
    print(f"Score: {report.score:.4f}")
    print(f"Threshold: {report.threshold:.4f}")
    print(f"Affected features: {', '.join(report.features)}")

# Performance metrics
print("\nPerformance metrics:")
print(results_drift.performance_metrics)