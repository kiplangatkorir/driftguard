import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from driftmonitor.drift_monitor_wrapper import DriftMonitorWrapper

# Load example dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Initialize drift monitoring
monitor = DriftMonitorWrapper(
    model=model,
    reference_data=X_train,
    alert_email="korirg543@gmail.com",  
    monitor_name="Iris Classifier Monitor"
)

# Function to simulate some drift
def add_noise(data, noise_level=0.5):
    return data + np.random.normal(0, noise_level, data.shape)

# Example monitoring usage
if __name__ == "__main__":
    # Monitor normal data
    print("\nMonitoring normal data...")
    results = monitor.monitor(X_test, y_test)
    print(f"Drift detected: {results['has_drift']}")
    print(f"Performance: {results['performance']}")
    
    # Monitor data with drift
    print("\nMonitoring data with drift...")
    X_test_drift = pd.DataFrame(
        add_noise(X_test.values, noise_level=1.0),
        columns=X_test.columns
    )
    results = monitor.monitor(X_test_drift, y_test)
    print(f"Drift detected: {results['has_drift']}")
    if results['has_drift']:
        print("Drift detected in features:", results['drift_detected_in'])
    print(f"Performance: {results['performance']}")
    
    # Get monitoring statistics
    stats = monitor.get_monitoring_stats()
    print("\nMonitoring Statistics:")
    print(f"Total Alerts: {stats['alerts']['total_alerts']}")
    print(f"Performance History: {stats['performance_history']}")