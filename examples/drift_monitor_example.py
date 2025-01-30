import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from driftmonitor import DriftMonitorWrapper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftMonitor")

# Step 1: Initialize the Model and Data
# For simplicity, we'll use a small sample dataset from sklearn
from sklearn.datasets import make_classification
X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
X_test, y_test = make_classification(n_samples=10, n_features=5, random_state=43)

# Train a simple model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 2: Initialize the DriftMonitorWrapper
monitor = DriftMonitorWrapper(
    model=model,  # The trained model
    reference_data=X_train,  # Training data to check drift against
    alert_email="korirg543@gmail.com",  # Email for drift alerts
    alert_threshold=0.5,  # Drift threshold
)

# Step 3: Simulate New Data (e.g., production data)
# Let's simulate some new data (e.g., data coming from production)
new_data = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

# Step 4: Monitor for Drift in the New Data
monitor_results = monitor.monitor(new_data)

# Step 5: Take Action Based on Monitoring Results
if monitor_results["has_drift"]:
    logger.warning(f"Drift detected in features: {monitor_results['drift_detected_in']}")
else:
    logger.info("No significant drift detected")

# Step 6: Optionally make predictions (if no drift is detected)
if not monitor_results["has_drift"]:
    predictions = model.predict(new_data)
    logger.info(f"Predictions: {predictions}")
else:
    logger.warning("Drift detected, predictions may not be reliable.")

