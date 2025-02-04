import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from datetime import datetime
import os
import json
import tempfile
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification

from driftguard.model_monitor import ModelMonitor  

@pytest.fixture
def dummy_binary_data():
    """Fixture to create simple binary classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def dummy_model():
    """Fixture to create a simple classifier."""
    model = DummyClassifier(strategy="stratified", random_state=42)
    return model

@pytest.fixture
def fitted_model(dummy_model, dummy_binary_data):
    """Fixture to create a fitted classifier."""
    X, y = dummy_binary_data
    dummy_model.fit(X[:80], y[:80])
    return dummy_model

@pytest.fixture
def test_data(dummy_binary_data):
    """Fixture to create test data."""
    X, y = dummy_binary_data
    return X[80:], y[80:]

@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

class TestModelMonitor:
    def test_init_with_valid_single_metric(self, fitted_model):
        """Test initialization with a single valid metric."""
        monitor = ModelMonitor(fitted_model, metrics="accuracy")
        assert monitor.metrics == ["accuracy"]
        assert monitor.threshold == 0.1
        assert monitor.save_dir is None

    def test_init_with_valid_multiple_metrics(self, fitted_model):
        """Test initialization with multiple valid metrics."""
        metrics = ["accuracy", "precision", "recall"]
        monitor = ModelMonitor(fitted_model, metrics=metrics)
        assert monitor.metrics == metrics

    def test_init_with_invalid_metric(self, fitted_model):
        """Test initialization with an invalid metric."""
        with pytest.raises(ValueError) as exc_info:
            ModelMonitor(fitted_model, metrics="invalid_metric")
        assert "Unsupported metric" in str(exc_info.value)

    def test_track_performance_basic(self, fitted_model, test_data):
        """Test basic performance tracking functionality."""
        monitor = ModelMonitor(fitted_model, metrics="accuracy")
        X_test, y_test = test_data
        
        results = monitor.track_performance(X_test, y_test)
        
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert isinstance(results["accuracy"], float)
        assert 0 <= results["accuracy"] <= 1

    def test_track_performance_multiple_metrics(self, fitted_model, test_data):
        """Test tracking multiple metrics."""
        metrics = ["accuracy", "precision", "recall"]
        monitor = ModelMonitor(fitted_model, metrics=metrics)
        X_test, y_test = test_data
        
        results = monitor.track_performance(X_test, y_test)
        
        assert all(metric in results for metric in metrics)
        assert all(isinstance(results[metric], float) for metric in metrics)

    def test_performance_history(self, fitted_model, test_data):
        """Test that performance history is properly maintained."""
        monitor = ModelMonitor(fitted_model, metrics="accuracy")
        X_test, y_test = test_data
        
        for _ in range(3):
            monitor.track_performance(X_test, y_test)
        
        assert len(monitor.performance_history) == 3
        assert all("timestamp" in record for record in monitor.performance_history)
        assert all("metrics" in record for record in monitor.performance_history)
        assert all("sample_size" in record for record in monitor.performance_history)

    def test_save_results(self, fitted_model, test_data, temp_dir):
        """Test saving results to disk."""
        monitor = ModelMonitor(
            fitted_model,
            metrics="accuracy",
            save_dir=temp_dir
        )
        X_test, y_test = test_data
        
        monitor.track_performance(X_test, y_test)
        
        filepath = os.path.join(temp_dir, "monitoring_history.json")
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        assert isinstance(saved_data, list)
        assert len(saved_data) == 1

    def test_get_summary_empty(self, fitted_model):
        """Test get_summary with no performance history."""
        monitor = ModelMonitor(fitted_model, metrics="accuracy")
        summary = monitor.get_summary()
        
        assert summary == {"status": "No performance data available"}

    def test_get_summary_with_data(self, fitted_model, test_data):
        """Test get_summary with performance history."""
        monitor = ModelMonitor(fitted_model, metrics=["accuracy", "precision"])
        X_test, y_test = test_data
        
        for _ in range(3):
            monitor.track_performance(X_test, y_test)
        
        summary = monitor.get_summary()
        
        assert "n_measurements" in summary
        assert summary["n_measurements"] == 3
        assert "time_span" in summary
        assert "metrics" in summary
        assert all(metric in summary["metrics"] 
                  for metric in ["accuracy", "precision"])
        
        for metric in ["accuracy", "precision"]:
            metric_stats = summary["metrics"][metric]
            assert all(stat in metric_stats 
                      for stat in ["mean", "std", "min", "max", "latest"])

    def test_performance_degradation_alert(self, fitted_model, test_data, capfd):
        """Test performance degradation alerting."""
        monitor = ModelMonitor(
            fitted_model,
            metrics="accuracy",
            threshold=0.001  
        )
        X_test, y_test = test_data
        
        monitor.track_performance(X_test, y_test)
        
        y_test_modified = y_test.copy()
        y_test_modified[0:5] = 1 - y_test_modified[0:5]  
        
        monitor.track_performance(X_test, y_test_modified)
        
        captured = capfd.readouterr()
        assert "Alert: accuracy has degraded" in captured.out

    def test_roc_auc_metric(self, fitted_model, test_data):
        """Test ROC AUC metric calculation."""
        monitor = ModelMonitor(fitted_model, metrics="roc_auc")
        X_test, y_test = test_data
        
        results = monitor.track_performance(X_test, y_test)
        
        assert "roc_auc" in results
        assert isinstance(results["roc_auc"], float)
        assert 0 <= results["roc_auc"] <= 1

    @pytest.mark.parametrize("sample_size", [10, 50, 100])
    def test_different_sample_sizes(self, fitted_model, dummy_binary_data, sample_size):
        """Test monitoring with different sample sizes."""
        monitor = ModelMonitor(fitted_model, metrics="accuracy")
        X, y = dummy_binary_data
        
        results = monitor.track_performance(X[:sample_size], y[:sample_size])
        
        assert monitor.performance_history[-1]["sample_size"] == sample_size