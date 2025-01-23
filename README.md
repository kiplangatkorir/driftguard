
# **Drift Detection Library**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
**Version:** 0.1.0  

## **Overview**
The Drift Detection Library is a Python package that empowers data scientists and machine learning engineers to monitor, detect, and respond to **data drift** and **concept drift** in real-time. These changes can negatively impact the performance of machine learning models in production. This library provides theoretical and practical tools to understand and mitigate drift, ensuring the reliability of deployed systems.

## **Theoretical Background**

### **1. What is Data Drift?**
Data drift refers to the change in the statistical properties of input data over time. It can occur due to:
- **Feature Drift**: Individual features in the dataset change distributions.
- **Concept Drift**: The relationship between input features and the target variable changes.

Such shifts can degrade model performance, as the model was trained on data that no longer represents the current data distribution.


### **2. Statistical Tests for Drift Detection**
The library relies on well-established statistical methods to detect drift:

#### **Kolmogorov-Smirnov (KS) Test**  
- **Used for Numerical Features**
- Measures the maximum difference between two cumulative distribution functions (CDFs) of the reference and new datasets.
- Null Hypothesis: The two datasets come from the same distribution.
- Result: A **p-value** that indicates whether the null hypothesis can be rejected.

#### **Chi-Square Test**
- **Used for Categorical Features**
- Compares observed vs. expected frequencies of categories in the reference and new datasets.
- Null Hypothesis: The two datasets have the same distribution of categorical values.
- Result: A **p-value** indicating whether the observed differences are significant.

#### **Drift Severity Score**
- Aggregates feature-level drift into an overall dataset-level drift score.
- Ranges between 0 (no drift) and 1 (severe drift).

### **3. Monitoring and Alerts**
The library builds on drift detection to:
1. Quantify **how much** the dataset has shifted.
2. Automatically **trigger alerts** when drift surpasses a user-defined threshold.
3. Suggest **actionable steps** for retraining or adjusting the model.

---

## **Cite This Work**
If you use the Drift Detection Library in your research or projects, please cite it as follows:

```bibtex
@software{korir2025driftmonitor,
  author = {Kiplangat Korir},
  title = {Drift Detection Library: A Python Library for Monitoring Data and Concept Drift in Machine Learning},
  year = {2025},
  url = {https://github.com/kiplangatkorir/driftmonitor},
  version = {0.1.0},
  license = {MIT}
}
```

Alternatively, include this text in your work:
> Korir, Kiplangat. (2025). *Drift Detection Library: A Python Library for Monitoring Data and Concept Drift in Machine Learning*. Version 0.1.0. Available at: https://github.com/kiplangatkorir/driftmonitor.

## **Features**
- **Drift Detection**:
  - Numerical drift detection using the Kolmogorov-Smirnov test.
  - Categorical drift detection using the Chi-Square test.
  - Drift severity scoring.

- **Drift Monitoring**:
  - Real-time drift monitoring with alerts.
  - Comprehensive drift reports.

- **Model Monitoring**:
  - Tracks model performance metrics (e.g., accuracy, latency).
  - Monitors consistency between inputs and outputs.

- **Version Management**:
  - Logs and manages model versions for traceability.

## **Installation**
Install the library using pip:

```bash
pip install driftmonitor
```

## **Quickstart Guide**

### **1. Initialize the Drift Detector**
```python
from driftmonitor.drift_detector import DriftDetector

# Load reference data (e.g., training data)
reference_data = load_reference_data()

# Initialize the drift detector
detector = DriftDetector(reference_data=reference_data)
```

### **2. Detect Drift in New Data**
```python
# Load new data (e.g., production data)
new_data = load_new_data()

# Run drift detection
drift_report = detector.detect_drift(new_data)

# Print drift summary
print(drift_report)
```

### **3. Real-Time Alerts**
```python
from driftmonitor.alert_manager import AlertManager

# Initialize the alert manager
alert_manager = AlertManager()

# Trigger an alert if drift severity exceeds 0.7
if drift_report['drift_severity'] > 0.7:
    alert_manager.send_alert("Significant drift detected!")
```

## **How It Works**
1. **Compare Distributions**:
   - Numerical features are tested using the Kolmogorov-Smirnov test.
   - Categorical features are tested using the Chi-Square test.

2. **Compute Drift Scores**:
   - Each feature's drift is scored and aggregated into a dataset-level severity score.

3. **Generate a Report**:
   - The drift report includes per-feature drift scores, dataset drift severity, and recommendations.

## **Directory Structure**
```
driftmonitor/                  # Root directory
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
├── examples/                  # Example scripts
│   ├── simple_example.py
│   └── advanced_example.py
├── LICENSE                    # License file
├── README.md                  # Library documentation
├── setup.py                   # Installation script
├── requirements.txt           # Dependencies
└── .gitignore                 # Ignore unnecessary files
```

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

## **License**
This project is licensed under the [MIT License](LICENSE).

## **Contact**
If you have questions or feedback, reach out:  
- **Email**: [korir@GraphFusion.onmicrosoft.com](mailto:korirkiplangat22@gmail.com)  
- **GitHub**: [kiplangatkorir](https://github.com/kiplangatkorir)

