
# **Drift Guard**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
**Version:** 0.1.0  

**DriftGuard** is a simple, flexible, and effective library for detecting **data drift** and **concept drift** in your machine learning models. It helps you keep track of model performance in production and catch problems early, so you can fix them before they impact your results.

## Why DriftGuard?  
Models are powerful, but they aren't immune to change. Data drift and concept drift are inevitable as the world changes around you. DriftMonitor helps by automatically monitoring your model's predictions and data features to ensure they're still working as expected.  

**Concept drift** occurs when the underlying relationship between input data and predictions changes over time. **Data drift** happens when the distribution of your input data shifts, which can degrade model performance.  

Detecting this drift before it hurts your model is crucial for maintaining the accuracy of your predictions.

## Features  
- **Real-Time & Batch Monitoring**: Monitor your model's performance in real time or in batch mode.  
- **Automatic Drift Detection**: Instantly detects feature and concept drift.  
- **Performance Tracking**: Keeps track of performance metrics, so you know when your model is underperforming.  
- **Alert System**: Get email alerts whenever drift is detected.  
- **Easy to Integrate**: No complicated setup – just plug it into your existing machine learning pipeline.  

## Install  
To install DriftGuard, simply run:

```bash
pip install driftguard
```

## Usage  

Here’s a simple example of how to use DriftGuard with a model.

### Example: Monitoring Model Performance

```python
from driftguard import Wrapper
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Initialize DriftMonitor
monitor = Wrapper(
    model=model,
    reference_data=X_train,
    alert_email="alerts@company.com",
    monitor_name="Iris Model Monitor"
)

# Monitor new data
results = monitor.monitor(X_test, y_test)

print(f"Drift Detected: {results['has_drift']}")
print(f"Performance: {results['performance']}")
```

It’s that simple. You can now monitor how well your model performs over time and be alerted if something goes wrong.

## The Theory Behind Drift Detection  

### Concept Drift
When the relationship between inputs and outputs changes over time, that’s concept drift. Imagine you have a model that predicts house prices, but after a while, the factors that drive those prices shift. Concept drift happens when the model's understanding of what affects price changes as well.

### Data Drift  
Data drift is when the distribution of the input data changes. For example, if your model is trained on customer data from the last year, but this year’s data has a slightly different distribution, this is data drift. DriftMonitor catches that shift and lets you know when it happens.

### Detection Methods  
- **KS Test**: A statistical test to compare distributions of features between two datasets.  
- **JSD**: Measures how similar two probability distributions are.  
- **PSI**: Used for categorical and continuous features, helps track distribution stability.

DriftMonitor uses these techniques (and more) to detect when your model or data is drifting.

## Integration Examples  

### API-Based Monitoring (FastAPI Example)  

```python
from fastapi import FastAPI
import pandas as pd
from driftguard import Wrapper

app = FastAPI()

# Initialize DriftMonitor
drift_monitor = Wrapper(
    model=trained_model,
    reference_data=training_data,
    alert_email="ml-team@company.com"
)

@app.post("/predict")
async def predict(data: dict):
    input_data = pd.DataFrame([data])
    monitor_results = drift_monitor.monitor(input_data)
    prediction = trained_model.predict(input_data)[0]
    
    return {
        "prediction": prediction,
        "drift_detected": monitor_results['has_drift'],
        "drift_scores": monitor_results['drift_scores']
    }
```
This is just one way you can use DriftGuard. It fits easily into your workflow, whether you're working with batch processing, real-time APIs, or other machine learning pipelines.

## Contributing  
Want to contribute? Awesome! Here’s how you can help:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request!
## License  
DriftGuard is licensed under the MIT License. You can find the full text in the [LICENSE](LICENSE) file.

## Cite This Work  
If you use DriftGuard in your research, please cite it like this:

```bibtex
@software{korir2025driftmonitor,
  author = {Kiplangat Korir},
  title = {Drift Guard: A Python Library for Monitoring Data and Concept Drift in Machine Learning},
  year = {2025},
  url = {https://github.com/kiplangatkorir/driftguard},
  version = {0.1.0},
  license = {MIT}
}
```

Or include this text:
> Korir, Kiplangat. (2025). *Drift Guard: A Python Library for Monitoring Data and Concept Drift in Machine Learning*. Version 0.1.0. Available at: https://github.com/kiplangatkorir/driftguard.

## Contact  
Have any questions? Want to give feedback? Reach out to me at:

- **Email**: [korir@GraphFusion.onmicrosoft.com](mailto:korirkiplangat22@gmail.com)  
- **GitHub**: [kiplangatkorir](https://github.com/kiplangatkorir)  


