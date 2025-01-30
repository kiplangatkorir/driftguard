import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from driftmonitor.drift_detector import DriftDetector
from driftmonitor.model_monitor import ModelMonitor
from driftmonitor.alert_manager import AlertManager

# Mock the main class and its dependencies
@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def reference_data():
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })

@pytest.fixture
def new_data():
    return pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 50),  # Shifted distribution
        'feature2': np.random.normal(0, 1, 50)
    })

@pytest.fixture
def drift_monitor(mock_model, reference_data):
    with patch('driftmonitor.drift_detector.DriftDetector') as mock_detector, \
         patch('driftmonitor.model_monitor.ModelMonitor') as mock_monitor, \
         patch('driftmonitor.alert_manager.AlertManager') as mock_alert:
        
        monitor = DriftMonitorWrapper(
            model=mock_model,
            reference_data=reference_data,
            alert_email="test@example.com",
            alert_threshold=0.5,
            monitor_name="Test Monitor"
        )
        yield monitor

def test_initialization(mock_model, reference_data):
    """Test proper initialization of DriftMonitorWrapper"""
    monitor = DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data,
        alert_email="test@example.com"
    )
    
    assert monitor.model == mock_model
    assert monitor.reference_data.equals(reference_data)
    assert isinstance(monitor.model_monitor, ModelMonitor)
    assert isinstance(monitor.drift_detector, DriftDetector)
    assert isinstance(monitor.alert_manager, AlertManager)

def test_initialization_without_email(mock_model, reference_data):
    """Test initialization without alert email"""
    monitor = DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data
    )
    assert monitor.alert_manager.threshold == 0.5  # Default threshold

def test_monitor_no_drift(drift_monitor, new_data):
    """Test monitoring when no drift is detected"""
    drift_monitor.drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.2, 'p_value': 0.8},
        'feature2': {'drift_score': 0.1, 'p_value': 0.9}
    }
    
    results = drift_monitor.monitor(new_data)
    
    assert not results['has_drift']
    assert len(results['drift_detected_in']) == 0
    assert 'feature1' in results['drift_scores']
    assert 'feature2' in results['drift_scores']

def test_monitor_with_drift(drift_monitor, new_data):
    """Test monitoring when drift is detected"""
    drift_monitor.drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.7, 'p_value': 0.01},
        'feature2': {'drift_score': 0.2, 'p_value': 0.8}
    }
    
    results = drift_monitor.monitor(new_data)
    
    assert results['has_drift']
    assert 'feature1' in results['drift_detected_in']
    assert 'feature2' not in results['drift_detected_in']

def test_monitor_with_labels(drift_monitor, new_data):
    """Test monitoring with actual labels provided"""
    drift_monitor.drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.2, 'p_value': 0.8}
    }
    
    actual_labels = np.random.randint(0, 2, 50)
    mock_performance = {'accuracy': 0.95, 'f1_score': 0.94}
    drift_monitor.model_monitor.track_performance.return_value = mock_performance
    
    results = drift_monitor.monitor(new_data, actual_labels)
    
    assert results['performance'] == mock_performance
    assert drift_monitor.model_monitor.track_performance.called

def test_monitor_raise_on_drift(drift_monitor, new_data):
    """Test that monitor raises exception when configured"""
    drift_monitor.drift_detector.detect_drift.return_value = {
        'feature1': {'drift_score': 0.7, 'p_value': 0.01}
    }
    
    with pytest.raises(ValueError, match="Data drift detected above threshold"):
        drift_monitor.monitor(new_data, raise_on_drift=True)

def test_get_monitoring_stats(drift_monitor):
    """Test retrieving monitoring statistics"""
    mock_alert_stats = {'total_alerts': 5, 'last_alert': '2024-01-30'}
    mock_performance_history = [{'accuracy': 0.95}, {'accuracy': 0.93}]
    
    drift_monitor.alert_manager.get_alert_statistics.return_value = mock_alert_stats
    drift_monitor.model_monitor.performance_history = mock_performance_history
    
    stats = drift_monitor.get_monitoring_stats()
    
    assert stats['alerts'] == mock_alert_stats
    assert stats['performance_history'] == mock_performance_history

def test_invalid_email_configuration(mock_model, reference_data):
    """Test handling of invalid email configuration"""
    with patch('driftmonitor.alert_manager.AlertManager.set_recipient_email') as mock_set_email:
        mock_set_email.side_effect = ValueError("Invalid email")
        
        # Should not raise exception, just log warning
        monitor = DriftMonitorWrapper(
            model=mock_model,
            reference_data=reference_data,
            alert_email="invalid@email"
        )
        
        mock_set_email.assert_called_once()