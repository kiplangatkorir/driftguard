"""
Tests for DriftGuard functionality.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import yaml
from pathlib import Path

from driftguard.core.config import ModelConfig, DriftConfig
from driftguard.core.guardian import DriftGuard
from driftguard.core.drift import KSTestDriftDetector, JSDDriftDetector, PSIDriftDetector
from driftguard.core.monitor import ModelMonitor
from driftguard.core.validation import DataValidator

@pytest.fixture
def config():
    """Load test configuration"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def classification_data():
    """Generate classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    )
    y = pd.Series(y, name="target")
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture
def regression_data():
    """Generate regression dataset"""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    )
    y = pd.Series(y, name="target")
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture
def classification_model():
    """Create classification model"""
    return RandomForestClassifier(n_estimators=100, random_state=42)

@pytest.fixture
def regression_model():
    """Create regression model"""
    return RandomForestRegressor(n_estimators=100, random_state=42)

def test_drift_detection_classification(
    config,
    classification_data,
    classification_model
):
    """Test drift detection for classification"""
    X_train, X_test, y_train, y_test = classification_data
    
    # Train model
    model = classification_model
    model.fit(X_train, y_train)
    
    # Initialize DriftGuard
    drift_guard = DriftGuard(
        model_config=ModelConfig(**config['model']),
        drift_config=DriftConfig(**config['drift']),
        storage_config=config['storage']
    )
    
    drift_guard.initialize(model, X_train, y_train)
    
    # Test with normal data
    results = drift_guard.monitor(X_test[:100], y_test[:100])
    assert not results['drift_detected']
    assert results['validation_passed']
    assert len(results['drift_reports']) > 0
    assert len(results['metric_reports']) > 0
    
    # Test with drifted data
    drifted_data = X_test.copy()
    drifted_data['feature_0'] = drifted_data['feature_0'] + 5
    results = drift_guard.monitor(drifted_data[:100], y_test[:100])
    assert results['drift_detected']
    assert results['validation_passed']

def test_drift_detection_regression(config, regression_data, regression_model):
    """Test drift detection for regression"""
    X_train, X_test, y_train, y_test = regression_data
    
    # Train model
    model = regression_model
    model.fit(X_train, y_train)
    
    # Initialize DriftGuard with regression config
    config['model']['type'] = 'regression'
    config['model']['metrics'] = ['mse', 'rmse', 'mae']
    
    drift_guard = DriftGuard(
        model_config=ModelConfig(**config['model']),
        drift_config=DriftConfig(**config['drift']),
        storage_config=config['storage']
    )
    
    drift_guard.initialize(model, X_train, y_train)
    
    # Test with normal data
    results = drift_guard.monitor(X_test[:100], y_test[:100])
    assert not results['drift_detected']
    assert results['validation_passed']
    
    # Test with drifted data
    drifted_data = X_test.copy()
    drifted_data['feature_0'] = drifted_data['feature_0'] * 2
    results = drift_guard.monitor(drifted_data[:100], y_test[:100])
    assert results['drift_detected']

def test_data_validation(config, classification_data):
    """Test data validation"""
    X_train, X_test, y_train, y_test = classification_data
    
    validator = DataValidator(
        max_missing_pct=config['model']['max_missing_pct']
    )
    validator.initialize(X_train)
    
    # Test with valid data
    result = validator.validate(X_test)
    assert result.is_valid
    assert len(result.errors) == 0
    
    # Test with invalid data
    invalid_data = X_test.copy()
    invalid_data.iloc[0, 0] = np.nan  # Add missing value
    result = validator.validate(invalid_data)
    assert result.is_valid  # Single missing value should be okay
    
    # Add too many missing values
    invalid_data.iloc[:50, 0] = np.nan
    result = validator.validate(invalid_data)
    assert not result.is_valid
    assert len(result.errors) > 0

def test_model_monitoring(config, classification_data, classification_model):
    """Test model performance monitoring"""
    X_train, X_test, y_train, y_test = classification_data
    
    # Train model
    model = classification_model
    model.fit(X_train, y_train)
    
    monitor = ModelMonitor(
        model_type='classification',
        metrics=config['model']['metrics']
    )
    monitor.initialize(model, X_train)
    
    # Test performance tracking
    reports = monitor.track_performance(X_test, y_test)
    assert len(reports) > 0
    
    for report in reports:
        assert report.metric_name in config['model']['metrics']
        assert 0 <= report.value <= 1  # Classification metrics should be in [0,1]

def test_drift_detectors():
    """Test individual drift detectors"""
    # Generate test data
    np.random.seed(42)
    reference = pd.DataFrame({
        'A': np.random.normal(0, 1, 1000),
        'B': np.random.uniform(-1, 1, 1000)
    })
    
    # Test KS Test detector
    ks_detector = KSTestDriftDetector(DriftConfig(**{
        'method': 'ks_test',
        'threshold': 0.05
    }))
    ks_detector.initialize(reference)
    
    # No drift
    normal_data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.uniform(-1, 1, 100)
    })
    reports = ks_detector.detect_drift(normal_data)
    assert not any(r.has_drift for r in reports)
    
    # With drift
    drift_data = pd.DataFrame({
        'A': np.random.normal(2, 1, 100),  # Mean shift
        'B': np.random.uniform(-1, 1, 100)
    })
    reports = ks_detector.detect_drift(drift_data)
    assert any(r.has_drift for r in reports)
    
    # Test JSD detector
    jsd_detector = JSDDriftDetector(DriftConfig(**{
        'method': 'jsd',
        'threshold': 0.1
    }))
    jsd_detector.initialize(reference)
    
    reports = jsd_detector.detect_drift(normal_data)
    assert not any(r.has_drift for r in reports)
    
    reports = jsd_detector.detect_drift(drift_data)
    assert any(r.has_drift for r in reports)
    
    # Test PSI detector
    psi_detector = PSIDriftDetector(DriftConfig(**{
        'method': 'psi',
        'threshold': 0.2
    }))
    psi_detector.initialize(reference)
    
    reports = psi_detector.detect_drift(normal_data)
    assert not any(r.has_drift for r in reports)
    
    reports = psi_detector.detect_drift(drift_data)
    assert any(r.has_drift for r in reports)

def test_error_handling(config, classification_data, classification_model):
    """Test error handling"""
    X_train, X_test, y_train, y_test = classification_data
    
    # Test initialization with invalid data
    drift_guard = DriftGuard(
        model_config=ModelConfig(**config['model']),
        drift_config=DriftConfig(**config['drift']),
        storage_config=config['storage']
    )
    
    # Try to monitor before initialization
    with pytest.raises(ValueError):
        drift_guard.monitor(X_test)
    
    # Initialize properly
    drift_guard.initialize(classification_model, X_train, y_train)
    
    # Try to monitor with missing features
    invalid_data = X_test.drop(columns=['feature_0'])
    with pytest.raises(ValueError):
        drift_guard.monitor(invalid_data, raise_on_drift=True)
    
    # Try to monitor with wrong data types
    invalid_data = X_test.astype(str)
    with pytest.raises(ValueError):
        drift_guard.monitor(invalid_data, raise_on_drift=True)
