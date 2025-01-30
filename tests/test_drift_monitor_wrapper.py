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

from driftmonitor.drift_monitor_wrapper import DriftMonitorWrapper

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def reference_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

@pytest.fixture
def new_data():
    return pd.DataFrame({
        'feature1': [1.1, 2.2, 3.3, 4.4, 5.5],
        'feature2': [15, 25, 35, 45, 55]
    })

@pytest.fixture
def actual_labels():
    return np.array([0, 1, 0, 1, 0])

@pytest.fixture
def drift_monitor(mock_model, reference_data):
    return DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data,
        alert_email="test@example.com",
        alert_threshold=0.5,
        monitor_name="Test Monitor"
    )

def test_initialization(mock_model, reference_data):
    """Test successful initialization of DriftMonitorWrapper"""
    monitor = DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data,
        monitor_name="Test Monitor"
    )
    
    assert monitor.model == mock_model
    assert monitor.reference_data.equals(reference_data)
    assert monitor.monitor_name == "Test Monitor"
    assert isinstance(monitor.model_monitor, ModelMonitor)
    assert isinstance(monitor.drift_detector, DriftDetector)
    assert isinstance(monitor.alert_manager, AlertManager)

def test_initialization_with_invalid_email(mock_model, reference_data):
    """Test initialization with invalid email configuration"""
    with patch('logging.getLogger') as mock_logger:
        monitor = DriftMonitorWrapper(
            model=mock_model,
            reference_data=reference_data,
            alert_email="invalid-email",
            monitor_name="Test Monitor"
        )
        
        mock_logger.return_value.warning.assert_called_once()

@pytest.mark.parametrize("drift_score,expected_drift", [
    (0.3, False),  # No drift
    (0.7, True)    # Drift detected
])
def test_monitor_drift_detection(
    drift_monitor,
    new_data,
    drift_score,
    expected_drift
):
    """Test drift detection with different drift scores"""
    mock_drift_report = {
        'feature1': {
            'drift_score': drift_score,
            'p_value': 0.05
        }
    }
    
    with patch.object(
        drift_monitor.drift_detector,
        'detect_drift',
        return_value=mock_drift_report
    ):
        results = drift_monitor.monitor(new_data)
        
        assert results['has_drift'] == expected_drift
        if expected_drift:
            assert 'feature1' in results['drift_detected_in']
        else:
            assert len(results['drift_detected_in']) == 0

def test_monitor_with_performance_tracking(
    drift_monitor,
    new_data,
    actual_labels
):
    """Test monitoring with performance tracking enabled"""
    mock_performance = {'accuracy': 0.95}
    
    with patch.object(
        drift_monitor.drift_detector,
        'detect_drift',
        return_value={}
    ), patch.object(
        drift_monitor.model_monitor,
        'track_performance',
        return_value=mock_performance
    ):
        results = drift_monitor.monitor(
            new_data,
            actual_labels=actual_labels
        )
        
        assert results['performance'] == mock_performance
        drift_monitor.model_monitor.track_performance.assert_called_once()

def test_monitor_raises_on_drift(drift_monitor, new_data):
    """Test that monitor raises exception when configured"""
    mock_drift_report = {
        'feature1': {
            'drift_score': 0.8,
            'p_value': 0.01
        }
    }
    
    with patch.object(
        drift_monitor.drift_detector,
        'detect_drift',
        return_value=mock_drift_report
    ), pytest.raises(ValueError, match="Data drift detected above threshold"):
        drift_monitor.monitor(new_data, raise_on_drift=True)

def test_get_monitoring_stats(drift_monitor):
    """Test retrieving monitoring statistics"""
    mock_alert_stats = {'total_alerts': 5}
    mock_performance_history = [{'accuracy': 0.95}]
    
    with patch.object(
        drift_monitor.alert_manager,
        'get_alert_statistics',
        return_value=mock_alert_stats
    ), patch.object(
        drift_monitor.model_monitor,
        'performance_history',
        mock_performance_history
    ):
        stats = drift_monitor.get_monitoring_stats()
        
        assert stats['alerts'] == mock_alert_stats
        assert stats['performance_history'] == mock_performance_history

def test_alert_manager_threshold(mock_model, reference_data):
    """Test alert threshold configuration"""
    custom_threshold = 0.75
    monitor = DriftMonitorWrapper(
        model=mock_model,
        reference_data=reference_data,
        alert_threshold=custom_threshold
    )
    
    assert monitor.alert_manager.threshold == custom_threshold

def test_monitor_multiple_feature_drift(drift_monitor, new_data):
    """Test monitoring multiple features with drift"""
    mock_drift_report = {
        'feature1': {
            'drift_score': 0.8,
            'p_value': 0.01
        },
        'feature2': {
            'drift_score': 0.3,
            'p_value': 0.2
        }
    }
    
    with patch.object(
        drift_monitor.drift_detector,
        'detect_drift',
        return_value=mock_drift_report
    ):
        results = drift_monitor.monitor(new_data)
        
        assert results['has_drift']
        assert 'feature1' in results['drift_detected_in']
        assert 'feature2' not in results['drift_detected_in']
        assert len(results['drift_scores']) == 2
        assert results['drift_scores']['feature1'] == 0.8
        assert results['drift_scores']['feature2'] == 0.3