import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from driftmonitor.drift_monitor_wrapper import DriftMonitorWrapper

@pytest.fixture
def mock_model():
    """Fixture for a mock model."""
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, 1])
    return model

@pytest.fixture
def reference_data():
    """Fixture for reference data."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

@pytest.fixture
def new_data():
    """Fixture for new data (with potential drift)."""
    return pd.DataFrame({
        'feature1': [6, 7, 8, 9, 10],
        'feature2': [60, 70, 80, 90, 100]
    })

@pytest.fixture
def monitor(mock_model, reference_data):
    """Fixture to create a DriftMonitorWrapper instance."""
    return DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data,
        alert_email="korirg543@gmail.com",
        alert_threshold=0.5
    )

def test_initialization(monitor, mock_model, reference_data):
    """Test the initialization of the DriftMonitorWrapper."""
    assert monitor.model == mock_model
    assert monitor.reference_data.equals(reference_data)
    assert monitor.alert_manager.threshold == 0.5
    assert monitor.alert_manager.recipient_email == "korirg543@gmail.com"
    
def test_monitor_drift_detection(monitor, new_data):
    """Test the drift detection functionality."""
    # Simulate the drift detection process
    drift_results = monitor.monitor(new_data=new_data)
    
    # Check if drift was detected
    assert drift_results['has_drift'] is True
    assert len(drift_results['drift_detected_in']) > 0
    assert 'drift_scores' in drift_results
    assert drift_results['drift_scores']['feature1'] > 0.5  # Assuming drift score is > 0.5 for feature1

def test_monitor_no_drift(monitor, reference_data):
    """Test the case where no drift is detected."""
    # Create new data that doesn't cause drift
    no_drift_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    
    drift_results = monitor.monitor(new_data=no_drift_data)
    
    # No drift should be detected
    assert drift_results['has_drift'] is False
    assert len(drift_results['drift_detected_in']) == 0

def test_monitor_performance(monitor, new_data):
    """Test the performance monitoring functionality."""
    actual_labels = pd.Series([1, 0, 1, 0, 1])  # Sample true labels
    
    # Simulate monitoring with performance tracking
    drift_results = monitor.monitor(new_data=new_data, actual_labels=actual_labels)
    
    # Check if performance data is returned
    assert drift_results['performance'] is not None
    assert 'accuracy' in drift_results['performance']
    assert 'precision' in drift_results['performance']

def test_alert_on_drift(monitor, new_data):
    """Test if alerting mechanism triggers on drift."""
    # Assuming threshold is set to 0.5 and drift score for feature1 is high
    drift_results = monitor.monitor(new_data=new_data)
    
    # Check if the alert manager was triggered (mocking email sending)
    monitor.alert_manager.check_and_alert.assert_called_once()
    
    # Verify that an alert was sent
    alert_call_args = monitor.alert_manager.check_and_alert.call_args[0][0]
    assert alert_call_args > 0.5  # Ensuring the alert was triggered due to drift
