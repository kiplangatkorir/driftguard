import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import unittest
import numpy as np
from sklearn.dummy import DummyClassifier
from driftmonitor.model_monitor import ModelMonitor

class TestModelMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple model and some test data."""
        # Create a dummy classifier model for testing
        self.model = DummyClassifier(strategy="most_frequent")
        self.model.fit([[0], [1], [2]], [0, 1, 1])  # Basic training
        self.monitor = ModelMonitor(model=self.model, metric="accuracy")
        
        # Example test data (3 samples, 1 feature each)
        self.X_test = np.array([[0], [1], [2]])
        self.y_test = np.array([0, 1, 1])
    
    def test_initialization(self):
        """Test that the ModelMonitor is initialized correctly."""
        self.assertEqual(self.monitor.model, self.model)
        self.assertEqual(self.monitor.metric, "accuracy")
        self.assertEqual(len(self.monitor.performance_history), 0)
    
    def test_track_performance(self):
        """Test that performance is tracked correctly."""
        results = self.monitor.track_performance(self.X_test, self.y_test)
        self.assertIn("accuracy", results)
        self.assertEqual(len(self.monitor.performance_history), 1)
        self.assertAlmostEqual(results["accuracy"], 1.0)
    
    def test_compute_metric(self):
        """Test that the metric computation works correctly."""
        accuracy = self.monitor._compute_metric(self.X_test, self.y_test)
        self.assertEqual(accuracy, 1.0)  # With DummyClassifier, the accuracy should be 1
    
    def test_empty_data(self):
        """Test how the monitor handles empty data."""
        empty_data = np.array([[], []])  # Empty data
        empty_labels = np.array([])
        results = self.monitor.track_performance(empty_data, empty_labels)
        self.assertEqual(results["accuracy"], 0.0)  # Should not break; accuracy 0.0 is expected

    def test_performance_history(self):
        """Test that performance history stores multiple values."""
        # Track performance once
        self.monitor.track_performance(self.X_test, self.y_test)
        # Track performance again with different data
        new_X = np.array([[0], [1], [1]])
        new_y = np.array([0, 1, 0])
        self.monitor.track_performance(new_X, new_y)
        
        # Ensure history contains two entries
        self.assertEqual(len(self.monitor.performance_history), 2)
    
    def test_invalid_metric(self):
        """Test that an error is raised for an unsupported metric."""
        # Try creating a ModelMonitor with an invalid metric
        with self.assertRaises(ValueError):
            ModelMonitor(self.model, metric="unsupported_metric")

if __name__ == "__main__":
    unittest.main()
