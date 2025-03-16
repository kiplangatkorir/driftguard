"""
Example script demonstrating DriftGuard functionality.
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pytest
from driftguard.core.guardian import DriftGuard
from driftguard.core.config import ConfigManager

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Split into reference and monitoring data
    X_ref, X_monitor, y_ref, y_monitor = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    return {
        'X_ref': X_ref,
        'X_monitor': X_monitor,
        'y_ref': y_ref,
        'y_monitor': y_monitor,
        'feature_names': feature_names
    }

@pytest.fixture
def sample_model(sample_data):
    """Create and train a sample model"""
    model = RandomForestClassifier(random_state=42)
    model.fit(sample_data['X_ref'], sample_data['y_ref'])
    return model

@pytest.fixture
def drift_guard(sample_data, sample_model):
    """Initialize DriftGuard instance"""
    reference_df = pd.DataFrame(
        sample_data['X_ref'],
        columns=sample_data['feature_names']
    )
    
    config_path = Path(__file__).parent / 'config.yaml'
    
    monitor = DriftGuard(
        model=sample_model,
        reference_data=reference_df,
        config_path=str(config_path),
        model_type="classification"
    )
    return monitor

@pytest.mark.asyncio
async def test_normal_monitoring(drift_guard, sample_data):
    """Test monitoring without drift"""
    # Prepare test data
    test_df = pd.DataFrame(
        sample_data['X_monitor'][:200],
        columns=sample_data['feature_names']
    )
    test_labels = sample_data['y_monitor'][:200]
    
    # Monitor batch
    results = await drift_guard.monitor_batch(
        test_df,
        test_labels,
        metadata={"test_type": "normal"}
    )
    
    # Verify results
    assert results['status'] == 'success'
    assert not results['drift_detected']
    assert 'performance_metrics' in results
    assert all(
        metric in results['performance_metrics']
        for metric in ['accuracy', 'f1', 'roc_auc']
    )

@pytest.mark.asyncio
async def test_drift_detection(drift_guard, sample_data):
    """Test monitoring with artificial drift"""
    # Prepare test data with drift
    test_data = sample_data['X_monitor'][200:400].copy()
    test_data[:, 0] += 2.0  # Add drift to first feature
    test_data[:, 1] *= 1.5  # Add drift to second feature
    
    test_df = pd.DataFrame(
        test_data,
        columns=sample_data['feature_names']
    )
    test_labels = sample_data['y_monitor'][200:400]
    
    # Monitor batch
    results = await drift_guard.monitor_batch(
        test_df,
        test_labels,
        metadata={"test_type": "drift"}
    )
    
    # Verify results
    assert results['status'] == 'success'
    assert results['drift_detected']
    assert any(
        report['drift_score'] > drift_guard.config_manager.config.drift.threshold
        for report in results['drift_reports']
    )

@pytest.mark.asyncio
async def test_performance_tracking(drift_guard, sample_data):
    """Test performance tracking functionality"""
    # Monitor multiple batches
    batch_size = 100
    for i in range(3):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        test_df = pd.DataFrame(
            sample_data['X_monitor'][start_idx:end_idx],
            columns=sample_data['feature_names']
        )
        test_labels = sample_data['y_monitor'][start_idx:end_idx]
        
        await drift_guard.monitor_batch(
            test_df,
            test_labels,
            metadata={"batch_id": f"batch_{i}"}
        )
    
    # Get monitoring summary
    summary = drift_guard.get_monitoring_summary()
    
    # Verify summary
    assert 'performance' in summary
    assert 'metrics' in summary['performance']
    for metric in ['accuracy', 'f1', 'roc_auc']:
        assert metric in summary['performance']['metrics']
        metric_data = summary['performance']['metrics'][metric]
        assert all(key in metric_data for key in ['current', 'mean', 'std'])

@pytest.mark.asyncio
async def test_data_validation(drift_guard, sample_data):
    """Test data validation"""
    # Test with invalid data
    invalid_df = pd.DataFrame(
        np.random.randn(100, 5),  # Wrong number of features
        columns=[f"wrong_feature_{i}" for i in range(5)]
    )
    
    # Monitor should handle invalid data gracefully
    results = await drift_guard.monitor_batch(
        invalid_df,
        np.random.randint(0, 2, 100),
        metadata={"test_type": "invalid"}
    )
    
    assert results['status'] == 'error'
    assert 'messages' in results

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
