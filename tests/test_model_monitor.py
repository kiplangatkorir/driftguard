import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import unittest
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from driftmonitor.model_monitor import ModelMonitor

class TestModelMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple dataset and dummy model for testing."""
        data = load_iris()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Dummy model that always predicts the most frequent class
        self.model = DummyClassifier(strategy="most_frequent")
        self.model.fit(self.X_train, self.y_train)

        # Initialize ModelMonitor
        self.monitor = ModelMonitor(model=self.model, metric="accuracy")

    def test_track_performance(self):
        """Test that performance is tracked correctly."""
        results = self.monitor.track_performance(self.X_test, self.y_test)
        self.assertIn("accuracy", results)
        self.assertAlmostEqual(results["accuracy"], 0.6666666666666666, places=7)  

    def test_compute_metric(self):
        """Test that the metric computation works correctly."""
        accuracy = self.monitor._compute_metric(self.X_test, self.y_test)
        self.assertAlmostEqual(accuracy, 0.6666666666666666, places=7)

    def test_empty_data(self):
        """Test how the monitor handles empty data."""
        empty_data = np.array([]).reshape(0, self.X_test.shape[1])  
        empty_labels = np.array([])
        
        with self.assertRaises(ValueError):
            self.monitor.track_performance(empty_data, empty_labels)

    def test_invalid_metric(self):
        """Test that an error is raised for an unsupported metric."""
        with self.assertRaises(ValueError):
            ModelMonitor(model=self.model, metric="unsupported_metric")

    def test_performance_history_tracking(self):
        """Test that performance history is updated after each tracking."""
        self.monitor.track_performance(self.X_test, self.y_test)
        self.monitor.track_performance(self.X_train, self.y_train)

        self.assertEqual(len(self.monitor.performance_history), 2)
        self.assertTrue(all(isinstance(value, float) for value in self.monitor.performance_history))

    def test_different_model_predictions(self):
        """Test that another classifier with a different strategy affects accuracy."""
        different_model = DummyClassifier(strategy="stratified")  # Random predictions
        different_model.fit(self.X_train, self.y_train)

        monitor_with_different_model = ModelMonitor(model=different_model, metric="accuracy")
        results = monitor_with_different_model.track_performance(self.X_test, self.y_test)
        
        self.assertIn("accuracy", results)
        self.assertNotEqual(results["accuracy"], self.monitor.track_performance(self.X_test, self.y_test)["accuracy"])

if __name__ == "__main__":
    unittest.main()
