
# **Drift Detection Library**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
**Version:** 0.1.0

## **Overview**
The Drift Detection Library is a Python package designed to monitor machine learning models in production by detecting **data drift** and **concept drift** in real-time. Data drift occurs when the statistical properties of data change over time, leading to degraded model performance. This library empowers machine learning engineers, data scientists, and businesses to stay ahead by detecting and responding to these changes effectively.

## **Features**
- **Drift Detection**:
  - Detects numerical feature drift using the **Kolmogorov-Smirnov test**.
  - Detects categorical feature drift using the **Chi-Square test**.
  - Computes drift scores for individual features and the dataset as a whole.
  
- **Drift Monitoring**:
  - Provides real-time alerts for significant drift.
  - Generates comprehensive drift reports.

- **Model Monitoring**:
  - Tracks key model metrics such as accuracy and latency over time.
  - Monitors input-output consistency.

- **Version Management**:
  - Logs and manages model versions for better traceability.


## **Why Use This Library?**
- **Ensure Model Performance**: Continuous drift monitoring prevents performance degradation in production.
- **Automated Alerts**: Receive notifications when drift is detected, so you can act immediately.
- **Simplify Model Retraining**: Gain insights into why your model is failing and identify which features to prioritize for retraining.
- **Data-Driven Decision Making**: Use detailed drift reports to inform business decisions.

## **Installation**
Install the library using pip:

```bash
pip install drift-detection-library
```

## **Quickstart Guide**

### **1. Initialize the Drift Detector**
```python
from driftmonitor.drift_detector import DriftDetector

# Load reference data (training data)
reference_data = load_reference_data()  # Replace with your data loading code

# Initialize the drift detector
detector = DriftDetector(reference_data=reference_data)
```

### **2. Detect Drift in New Data**
```python
# Load new data (production data)
new_data = load_new_data()  # Replace with your data loading code

# Run drift detection
drift_report = detector.detect_drift(new_data)

# Print drift summary
print(drift_report)
```

### **3. Receive Alerts for Significant Drift**
```python
from driftmonitor.alert_manager import AlertManager

# Initialize the alert manager
alert_manager = AlertManager()

# Trigger an alert if drift is significant
if drift_report['drift_severity'] > 0.7:
    alert_manager.send_alert("Significant drift detected in production data!")
```

## **How It Works**
The Drift Detection Library operates in three key steps:
1. **Compare Distributions**:
   - For **numerical features**, statistical tests (e.g., Kolmogorov-Smirnov) measure the difference between the distributions of reference data and new data.
   - For **categorical features**, Chi-Square tests assess the differences in observed vs. expected frequencies.

2. **Calculate Drift Scores**:
   - A drift score is computed for each feature, summarizing how much the feature has changed.

3. **Generate a Drift Report**:
   - The report includes:
     - Drift severity for individual features.
     - Overall dataset drift severity.
     - Suggested next steps (e.g., retraining the model).

## **Directory Structure**
```
driftmonitor/                  # Root directory of the library
├── driftmonitor/              # Library source code
│   ├── __init__.py            # Package initializer
│   ├── drift_detector.py      # Drift detection logic
│   ├── model_monitor.py       # Model monitoring logic
│   ├── alert_manager.py       # Alerting mechanism
│   ├── version_manager.py     # Model versioning
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
│   ├── test_drift_detector.py
│   ├── test_model_monitor.py
│   ├── test_alert_manager.py
│   └── test_version_manager.py
├── examples/                  # Example usage scripts
│   ├── simple_example.py
│   └── advanced_example.py
├── LICENSE                    # License file
├── README.md                  # Library documentation
├── setup.py                   # Installation script
├── requirements.txt           # List of dependencies
└── .gitignore                 # Ignore unnecessary files
```

## **Examples**

### Simple Drift Detection Example
```python
from driftmonitor.drift_detector import DriftDetector

# Reference data
reference_data = {"feature_1": [1, 2, 3, 4, 5], "feature_2": ["A", "B", "A", "C", "B"]}

# New data
new_data = {"feature_1": [6, 7, 8, 9, 10], "feature_2": ["C", "C", "A", "C", "C"]}

# Initialize detector and detect drift
detector = DriftDetector(reference_data)
drift_report = detector.detect_drift(new_data)

# View report
print(drift_report)
```

## **Contributing**
Contributions are welcome! Here's how you can help:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## **License**
This project is licensed under the [MIT License](LICENSE).

## **Contact**
For questions, feedback, or contributions, feel free to reach out:
- **Email**: korirkiplangat22@gmail.com

Let me know if you want to customize this further!
