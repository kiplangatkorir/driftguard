# DriftMonitor Documentation

## Overview
Drift Guard is a Python package that helps you monitor machine learning models in production for data drift and performance degradation. It provides automated monitoring, drift detection, and alerting capabilities with minimal setup required.

## Installation

```bash
pip install driftguard
```

## Quick Start

```python
from driftguard import Wrapper

# Initialize monitoring
monitor = Wrapper(
    model=your_model,
    reference_data=training_data,
    alert_email="your.email@company.com"
)

# Monitor new data
results = monitor.monitor(new_data, actual_labels)
```

## Core Components

### Wrapper
The main interface for monitoring your models. Combines drift detection, performance monitoring, and alerting in a single, easy-to-use package.

```python
from driftguard import Wrapper

monitor = Wrapper(
    model=trained_model,              # Your trained model
    reference_data=training_data,     # Reference data (usually training data)
    alert_email="user@company.com",   # Email for alerts (optional)
    alert_threshold=0.5,              # Drift threshold (default: 0.5)
    monitor_name="Production Model"   # Name for this monitor
)
```

#### Methods

##### monitor()
Monitor new data for drift and performance changes.

```python
results = monitor.monitor(
    new_data=new_data,           # New data to check
    actual_labels=true_labels,   # True labels (optional)
    raise_on_drift=False         # Whether to raise exception on drift
)
```

Returns:
```python
{
    'has_drift': bool,                    # Whether drift was detected
    'drift_detected_in': [str],           # List of features with drift
    'drift_scores': {                     # Drift scores per feature
        'feature_name': float,
    },
    'performance': {                      # If labels provided
        'accuracy': float
    }
}
```

##### get_monitoring_stats()
Get current monitoring statistics.

```python
stats = monitor.get_monitoring_stats()
```

Returns:
```python
{
    'alerts': {
        'total_alerts': int,
        'successful_alerts': int,
        'failed_alerts': int
    },
    'performance_history': [float]
}
```

### Individual Components

#### DriftDetector
Detects statistical drift between reference and new data.

```python
from driftguard.drift_detector import DriftDetector

detector = DriftDetector(reference_data=training_data)
drift_report = detector.detect_drift(new_data)
```

#### ModelMonitor
Tracks model performance over time.

```python
from driftguard.model_monitor import ModelMonitor

monitor = ModelMonitor(model)
performance = monitor.track_performance(data, labels)
```

#### AlertManager
Handles email alerts when drift or performance issues are detected.

```python
from driftguard.alert_manager import AlertManager

alerter = AlertManager(threshold=0.5)
alerter.set_recipient_email("user@company.com")
alerter.check_and_alert(drift_score=0.7, message="High drift detected!")
```

## Integration Examples

### Basic Production Pipeline
```python
from driftguard import Wrapper

def production_pipeline(new_data, actual_labels=None):
    # Initialize monitoring (do this once)
    monitor = Wrapper(
        model=production_model,
        reference_data=training_data,
        alert_email="alerts@company.com"
    )
    
    # Monitor new data
    results = monitor.monitor(new_data, actual_labels)
    
    # Handle results
    if results['has_drift']:
        logger.warning(f"Drift detected in features: {results['drift_detected_in']}")
    
    # Make predictions
    return production_model.predict(new_data)
```

### Batch Processing
```python
def batch_monitoring():
    monitor = Wrapper(
        model=model,
        reference_data=reference_data,
        alert_email="team@company.com"
    )
    
    while True:
        # Get new batch of data
        new_batch = get_new_data()
        
        # Monitor for drift
        results = monitor.monitor(new_batch)
        
        # Check monitoring stats
        if len(monitor.get_monitoring_stats()['alerts']) > 0:
            # Consider model retraining
            pass
            
        time.sleep(3600)  # Check hourly
```

## Configuration

### Email Alerts
The package uses SMTP for sending alerts. Configure these environment variables:

```bash
SENDER_EMAIL=your-system@company.com
SENDER_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Logging
The package uses Python's built-in logging. Configure it in your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Best Practices

1. **Reference Data**: Use a representative sample of your training data as reference data.
2. **Alert Thresholds**: Start with the default threshold (0.5) and adjust based on your needs.
3. **Performance Monitoring**: When possible, provide actual labels to monitor model performance.
4. **Regular Checks**: Set up regular monitoring intervals appropriate for your use case.
5. **Email Alerts**: Configure email alerts for critical models to get immediate notifications.

## Troubleshooting

Common issues and solutions:

1. **Email Configuration**
   - Ensure environment variables are set correctly
   - For Gmail, use App Password instead of regular password
   - Check SMTP server and port settings

2. **Performance Tracking**
   - Ensure labels match the model's expected format
   - Check that all features in reference data exist in new data

3. **Drift Detection**
   - Verify data types match between reference and new data
   - Ensure no missing values in critical features

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
