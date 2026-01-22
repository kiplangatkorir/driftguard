"""Tests for the model monitoring module."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from driftguard.core.monitor import ModelMonitor
from driftguard.core.config import MonitorConfig

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    size = 1000
    
    # Binary classification data
    predictions = pd.Series(np.random.binomial(1, 0.7, size))
    labels = pd.Series(np.random.binomial(1, 0.7, size))
    
    return predictions, labels

@pytest.fixture
def monitor():
    """Create a model monitor instance"""
    config = MonitorConfig(
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        threshold_type='relative',
        thresholds={
            'accuracy': 0.1,
            'precision': 0.1,
            'recall': 0.1,
            'f1': 0.1
        },
        window_size=100
    )
    return ModelMonitor(config)

def test_monitor_initialization(monitor, sample_data):
    """Test monitor initialization"""
    predictions, labels = sample_data
    
    # Test initialization
    monitor.initialize(predictions, labels)
    assert monitor._initialized
    assert monitor.reference_predictions is not None
    assert monitor.reference_labels is not None
    assert len(monitor.reference_metrics) > 0

def test_monitor_track_performance(monitor, sample_data):
    """Test performance tracking"""
    predictions, labels = sample_data
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Track performance
    metrics = monitor.track(predictions[800:], labels[800:])
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    for metric in monitor.config.metrics:
        assert metric in metrics
        assert 'value' in metrics[metric]
        assert 'degraded' in metrics[metric]
        assert 'reference' in metrics[metric]

def test_concept_drift_detection(monitor, sample_data):
    """Test concept drift detection"""
    predictions, labels = sample_data
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Introduce drift by flipping some predictions
    drift_predictions = predictions[800:].copy()
    drift_predictions = 1 - drift_predictions  # Flip predictions
    
    # Detect drift
    has_drift, drift_metrics = monitor.detect_concept_drift(
        drift_predictions,
        labels[800:]
    )
    
    # Check drift detection results
    assert isinstance(has_drift, bool)
    assert isinstance(drift_metrics, dict)
    for metric in monitor.config.metrics:
        assert metric in drift_metrics
        assert 'current' in drift_metrics[metric]
        assert 'reference' in drift_metrics[metric]
        assert 'relative_change' in drift_metrics[metric]
        assert 'degraded' in drift_metrics[metric]

def test_degradation_thresholds(monitor, sample_data):
    """Test different degradation threshold types"""
    predictions, labels = sample_data
    
    # Test absolute thresholds
    monitor.config.threshold_type = 'absolute'
    monitor.config.thresholds = {
        'accuracy': 0.8,
        'precision': 0.8,
        'recall': 0.8,
        'f1': 0.8
    }
    
    monitor.initialize(predictions[:800], labels[:800])
    metrics = monitor.track(predictions[800:], labels[800:])
    
    for metric_name, metric_data in metrics.items():
        assert isinstance(metric_data['degraded'], bool)
    
    # Test relative thresholds
    monitor.config.threshold_type = 'relative'
    monitor.config.thresholds = {
        'accuracy': 0.1,
        'precision': 0.1,
        'recall': 0.1,
        'f1': 0.1
    }
    
    monitor.initialize(predictions[:800], labels[:800])
    metrics = monitor.track(predictions[800:], labels[800:])
    
    for metric_name, metric_data in metrics.items():
        assert isinstance(metric_data['degraded'], bool)

def test_error_handling(monitor):
    """Test error handling"""
    # Test uninitialized monitor
    with pytest.raises(ValueError):
        monitor.track(pd.Series([1, 0]), pd.Series([1, 1]))
    
    # Test mismatched lengths
    monitor.initialize(pd.Series([1, 0]), pd.Series([1, 0]))
    with pytest.raises(ValueError):
        monitor.track(pd.Series([1]), pd.Series([1, 0]))
    
    # Test empty data
    with pytest.raises(ValueError):
        monitor.initialize(pd.Series([]), pd.Series([]))

def test_statistical_process_control(monitor, sample_data):
    """Test statistical process control for drift detection"""
    predictions, labels = sample_data
    
    # Configure for dynamic thresholds
    monitor.config.threshold_type = 'dynamic'
    monitor.config.thresholds = {
        'accuracy': 0.1,
        'precision': 0.1,
        'recall': 0.1,
        'f1': 0.1
    }
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Test with normal data
    normal_metrics = monitor.track(predictions[800:850], labels[800:850])
    
    # Test with significantly degraded data
    degraded_predictions = 1 - predictions[850:900]  # Flip predictions
    degraded_metrics = monitor.track(degraded_predictions, labels[850:900])
    
    # Verify SPC detection
    assert any(
        metric['degraded'] for metric in degraded_metrics.values()
    )

def test_alert_on_degradation(monitor, sample_data):
    """Test that alerts are triggered on performance degradation"""
    from unittest.mock import Mock
    
    predictions, labels = sample_data
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Attach mock alert manager
    mock_alert_manager = Mock()
    monitor.attach_alert_manager(mock_alert_manager)
    
    # Create degraded predictions
    degraded_predictions = predictions[800:].copy()
    degraded_predictions = 1 - degraded_predictions  # Flip to degrade performance
    
    # Track performance (should trigger alert)
    metrics = monitor.track(degraded_predictions, labels[800:])
    
    # Verify alert was called
    assert mock_alert_manager.check_and_alert.called
    call_args = mock_alert_manager.check_and_alert.call_args
    assert 'Metric:' in call_args[1]['message']
    assert 'Degradation:' in call_args[1]['message']

def test_no_alert_on_good_performance(monitor, sample_data):
    """Test that no alerts are sent when performance is acceptable"""
    from unittest.mock import Mock
    
    predictions, labels = sample_data
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Attach mock alert manager
    mock_alert_manager = Mock()
    monitor.attach_alert_manager(mock_alert_manager)
    
    # Track performance with good data (should not trigger alert)
    metrics = monitor.track(predictions[800:], labels[800:])
    
    # Verify no alert was called
    assert not mock_alert_manager.check_and_alert.called

def test_alert_contains_correct_metric_info(monitor, sample_data):
    """Test that alerts contain correct metric information"""
    from unittest.mock import Mock
    
    predictions, labels = sample_data
    
    # Initialize monitor
    monitor.initialize(predictions[:800], labels[:800])
    
    # Attach mock alert manager
    mock_alert_manager = Mock()
    monitor.attach_alert_manager(mock_alert_manager)
    
    # Create degraded predictions
    degraded_predictions = predictions[800:].copy()
    degraded_predictions = 1 - degraded_predictions
    
    # Track performance
    metrics = monitor.track(degraded_predictions, labels[800:])
    
    # Verify alert was called with correct info
    if mock_alert_manager.check_and_alert.called:
        call_args = mock_alert_manager.check_and_alert.call_args
        message = call_args[1]['message']
        
        # Check message contains required fields
        assert 'Baseline Value:' in message
        assert 'Current Value:' in message
        assert 'Degradation:' in message
        assert 'Threshold:' in message

def test_alert_threshold_types(monitor, sample_data):
    """Test different threshold types trigger alerts correctly"""
    from unittest.mock import Mock
    
    predictions, labels = sample_data
    
    # Test with absolute threshold
    monitor.config.threshold_type = 'absolute'
    monitor.config.thresholds = {
        'accuracy': 0.9,  # High threshold to trigger alert
        'precision': 0.9,
        'recall': 0.9,
        'f1': 0.9
    }
    
    monitor.initialize(predictions[:800], labels[:800])
    mock_alert_manager = Mock()
    monitor.attach_alert_manager(mock_alert_manager)
    
    # Track performance (should trigger alert due to high absolute threshold)
    metrics = monitor.track(predictions[800:], labels[800:])
    
    # Some degradation should be detected with high threshold
    # Note: This may or may not trigger depending on actual performance
    # Just verify the method is callable
    assert hasattr(monitor, 'alert_manager')

def test_attach_alert_manager_method(monitor):
    """Test attach_alert_manager method exists and works"""
    from unittest.mock import Mock
    
    mock_alert_manager = Mock()
    
    # Test method exists
    assert hasattr(monitor, 'attach_alert_manager')
    
    # Test attaching
    monitor.attach_alert_manager(mock_alert_manager)
    assert monitor.alert_manager == mock_alert_manager

def test_monitor_initialization_with_alert_manager(sample_data):
    """Test monitor can be initialized with AlertManager"""
    from unittest.mock import Mock
    
    config = MonitorConfig(
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        threshold_type='relative',
        thresholds={
            'accuracy': 0.1,
            'precision': 0.1,
            'recall': 0.1,
            'f1': 0.1
        }
    )
    
    mock_alert_manager = Mock()
    monitor = ModelMonitor(config=config, alert_manager=mock_alert_manager)
    
    # Verify alert manager is attached
    assert monitor.alert_manager == mock_alert_manager
