import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from driftmonitor.drift_detector import DriftDetector
from driftmonitor.model_monitor import ModelMonitor
from driftmonitor.alert_manager import AlertManager
from driftmonitor.drift_monitor_wrapper import DriftMonitorWrapper  

@pytest.fixture
def sample_data():
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100)
    })
    new_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 100),
        'feature2': np.random.normal(5, 2, 100)
    })
    actual_labels = np.random.randint(0, 2, 100)
    return reference_data, new_data, actual_labels

@pytest.fixture
def mock_components():
    mock_model = MagicMock()
    mock_drift_detector = MagicMock(spec=DriftDetector)
    mock_model_monitor = MagicMock(spec=ModelMonitor)
    mock_alert_manager = MagicMock(spec=AlertManager)
    return mock_model, mock_drift_detector, mock_model_monitor, mock_alert_manager

def test_initialization(mock_components, sample_data):
    model, _, _, _ = mock_components
    reference_data, _, _ = sample_data
    monitor = DriftMonitorWrapper(model, reference_data, alert_email="korirg543@gmail.com")
    assert monitor.model == model
    assert monitor.monitor_name == "Model Monitor"
    assert isinstance(monitor.drift_detector, DriftDetector)
    assert isinstance(monitor.model_monitor, ModelMonitor)
    assert isinstance(monitor.alert_manager, AlertManager)

def test_monitor_no_drift(mock_components, sample_data):
    model, drift_detector, model_monitor, alert_manager = mock_components
    reference_data, new_data, actual_labels = sample_data

    drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.2, 'p_value': 0.8},
        'feature2': {'drift_score': 0.1, 'p_value': 0.9}
    }
    model_monitor.track_performance.return_value = {'accuracy': 0.85}
    
    alert_manager.threshold = 0.5  # Fix: Add this to mock the threshold

    monitor = DriftMonitorWrapper(model, reference_data)
    monitor.drift_detector = drift_detector
    monitor.model_monitor = model_monitor
    monitor.alert_manager = alert_manager

    results = monitor.monitor(new_data, actual_labels)
    assert results["drift_detected"] is False  # Adjust assertion based on expected output


def test_monitor_with_drift(mock_components, sample_data):
    model, drift_detector, model_monitor, alert_manager = mock_components
    reference_data, new_data, actual_labels = sample_data
    
    drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.6, 'p_value': 0.01},
        'feature2': {'drift_score': 0.4, 'p_value': 0.05}
    }
    
    monitor = DriftMonitorWrapper(model, reference_data)
    monitor.drift_detector = drift_detector
    monitor.model_monitor = model_monitor
    monitor.alert_manager = alert_manager
    
    results = monitor.monitor(new_data)
    
    assert results['has_drift']
    assert 'feature1' in results['drift_detected_in']
    assert results['drift_scores']['feature1'] > 0.5
    alert_manager.check_and_alert.assert_called()

def test_monitor_raise_on_drift(mock_components, sample_data):
    model, drift_detector, model_monitor, alert_manager = mock_components
    reference_data, new_data, _ = sample_data
    
    drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.7, 'p_value': 0.01}
    }
    
    monitor = DriftMonitorWrapper(model, reference_data)
    monitor.drift_detector = drift_detector
    monitor.alert_manager = alert_manager
    
    with pytest.raises(ValueError, match="Data drift detected above threshold"):
        monitor.monitor(new_data, raise_on_drift=True)

def test_get_monitoring_stats(mock_components, sample_data):
    model, _, model_monitor, alert_manager = mock_components
    reference_data, _, _ = sample_data
    
    alert_manager.get_alert_statistics.return_value = {'alerts_sent': 3}
    model_monitor.performance_history = [{'accuracy': 0.85}]
    
    monitor = DriftMonitorWrapper(model, reference_data)
    monitor.alert_manager = alert_manager
    monitor.model_monitor = model_monitor
    
    stats = monitor.get_monitoring_stats()
    
    assert stats['alerts']['alerts_sent'] == 3
    assert stats['performance_history'][0]['accuracy'] == 0.85
