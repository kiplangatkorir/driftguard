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
