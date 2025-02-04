import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from driftguard.drift_detector import DriftDetector
from driftguard.model_monitor import ModelMonitor
from driftguard.alert_manager import AlertManager
from driftguard.wrapper import DriftMonitorWrapper

# Fixture for sample data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.binomial(1, 0.5, 100)
    })

# Fixture for mock model
@pytest.fixture
def mock_model():
    model = Mock()
    model.predict.return_value = np.random.binomial(1, 0.5, 100)
    return model

# Fixture for DriftMonitorWrapper instance
@pytest.fixture
def drift_monitor(sample_data, mock_model):
    return DriftMonitorWrapper(
        model=mock_model,
        reference_data=sample_data,
        alert_email="korirg543@gmail.com",
        alert_threshold=0.5,
        monitor_name="Test Monitor"
    )

def test_initialization(sample_data, mock_model):
    """Test proper initialization of DriftMonitorWrapper"""
    monitor = DriftMonitorWrapper(
        model=mock_model,
        reference_data=sample_data
    )
    
    assert monitor.model == mock_model
    assert monitor.reference_data.equals(sample_data)
    assert monitor.monitor_name == "Model Monitor"  
    assert isinstance(monitor.model_monitor, ModelMonitor)
    assert isinstance(monitor.drift_detector, DriftDetector)
    assert isinstance(monitor.alert_manager, AlertManager)

def test_initialization_with_invalid_email(sample_data, mock_model, caplog):
    """Test initialization with invalid email handling"""
    DriftMonitorWrapper(
        model=mock_model,
        reference_data=sample_data,
        alert_email="invalid-email"
    )
    assert "Invalid email configuration" in caplog.text

@patch('driftmonitor.drift_detector.DriftDetector.detect_drift')
def test_monitor_no_drift(mock_detect_drift, drift_monitor, sample_data):
    """Test monitoring when no drift is detected"""
    # Mock drift detector to return no drift
    mock_detect_drift.return_value = {
        'feature1': {'drift_score': 0.1, 'p_value': 0.8},
        'feature2': {'drift_score': 0.2, 'p_value': 0.7}
    }
    
    results = drift_monitor.monitor(sample_data)
    
    assert not results['has_drift']
    assert len(results['drift_detected_in']) == 0
    assert len(results['drift_scores']) == 2
    assert results['performance'] is None

@patch('driftmonitor.drift_detector.DriftDetector.detect_drift')
def test_monitor_with_drift(mock_detect_drift, drift_monitor, sample_data):
    """Test monitoring when drift is detected"""
    # Mock drift detector to return drift in feature1
    mock_detect_drift.return_value = {
        'feature1': {'drift_score': 0.8, 'p_value': 0.01},
        'feature2': {'drift_score': 0.2, 'p_value': 0.7}
    }
    
    results = drift_monitor.monitor(sample_data)
    
    assert results['has_drift']
    assert 'feature1' in results['drift_detected_in']
    assert len(results['drift_scores']) == 2
    assert results['drift_scores']['feature1'] == 0.8

def test_monitor_with_labels(drift_monitor, sample_data):
    """Test monitoring with actual labels provided"""
    actual_labels = np.random.binomial(1, 0.5, 100)
    
    results = drift_monitor.monitor(
        sample_data,
        actual_labels=actual_labels
    )
    
    assert 'performance' in results
    assert results['performance'] is not None

def test_monitor_raise_on_drift(drift_monitor, sample_data):
    """Test raise_on_drift flag behavior"""
    # Patch drift detector to always detect drift
    with patch('driftmonitor.drift_detector.DriftDetector.detect_drift') as mock_detect:
        mock_detect.return_value = {
            'feature1': {'drift_score': 0.8, 'p_value': 0.01}
        }
        
        with pytest.raises(ValueError, match="Data drift detected above threshold"):
            drift_monitor.monitor(sample_data, raise_on_drift=True)

def test_get_monitoring_stats(drift_monitor):
    """Test retrieval of monitoring statistics"""
    stats = drift_monitor.get_monitoring_stats()
    
    assert 'alerts' in stats
    assert 'performance_history' in stats

def test_empty_dataframe_handling(drift_monitor):
    """Test handling of empty DataFrame input"""
    empty_df = pd.DataFrame()
    
    with pytest.raises((ValueError, AssertionError), match="Empty DataFrame"):
        drift_monitor.monitor(empty_df)

def test_different_column_handling(drift_monitor, sample_data):
    """Test handling of DataFrame with different columns"""
    # Create a DataFrame with missing required columns
    different_df = pd.DataFrame({
        'new_feature': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    
    # Expect ValueError or AssertionError for mismatched columns
    with pytest.raises((ValueError, AssertionError), match="Column mismatch"):
        drift_monitor.monitor(different_df)
        
def test_missing_required_columns(drift_monitor, sample_data):
    """Test handling of DataFrame with missing required columns"""
    # Create DataFrame missing some columns from reference data
    missing_cols_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100)
    })
    
    with pytest.raises((ValueError, AssertionError), match="Missing required columns"):
        drift_monitor.monitor(missing_cols_df)