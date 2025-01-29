import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from driftmonitor.drift_detector import DriftDetector
from driftmonitor.model_monitor import ModelMonitor
from driftmonitor.alert_manager import AlertManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Example dataset (Iris dataset for simplicity)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Monitor model performance
monitor = ModelMonitor(model)
performance = monitor.track_performance(X_test, y_test)
print("Initial model performance:", performance)

# Drift detection logic (using simple feature drift detection)
drift_detector = DriftDetector(reference_data=X_train)
drift_report = drift_detector.detect_drift(X_test)
print("Drift detection report:", drift_report)

# Alert system (notify if drift is detected above threshold)
alert_manager = AlertManager(threshold=0.5)
for feature, report in drift_report.items():
    if report["drift_score"] > alert_manager.threshold:
        alert_manager.send_alert(f"Drift detected in feature '{feature}' with score: {report['drift_score']}")

