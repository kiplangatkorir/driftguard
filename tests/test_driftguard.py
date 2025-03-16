"""
Test suite for DriftGuard functionality.
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from driftguard.core.config import DriftGuardConfig
from driftguard.core.drift import KSTestDriftDetector, JSDDriftDetector, PSIDriftDetector
from driftguard.core.monitor import ModelMonitor
from driftguard.core.validation import DataValidator
from driftguard.core.state import StateManager
from driftguard.core.alerts import AlertManager

class TestDriftGuard(unittest.TestCase):
    """Test DriftGuard components"""
    
    def setUp(self):
        """Setup test environment"""
        # Load configuration
        self.config = DriftGuardConfig.from_yaml('tests/config.yaml')
        
        # Generate test data
        self._generate_test_data()
        
        # Initialize components
        self._initialize_components()
    
    def _generate_test_data(self):
        """Generate test datasets"""
        # Classification data
        X_cls, y_cls = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Split classification data
        (
            self.X_cls_train, self.X_cls_test,
            self.y_cls_train, self.y_cls_test
        ) = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
        
        # Create classification DataFrames
        self.cls_train_df = pd.DataFrame(
            self.X_cls_train,
            columns=[f'feature_{i}' for i in range(10)]
        )
        self.cls_test_df = pd.DataFrame(
            self.X_cls_test,
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        # Train classification model
        self.cls_model = RandomForestClassifier(random_state=42)
        self.cls_model.fit(self.X_cls_train, self.y_cls_train)
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        # Split regression data
        (
            self.X_reg_train, self.X_reg_test,
            self.y_reg_train, self.y_reg_test
        ) = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
        
        # Create regression DataFrames
        self.reg_train_df = pd.DataFrame(
            self.X_reg_train,
            columns=[f'feature_{i}' for i in range(10)]
        )
        self.reg_test_df = pd.DataFrame(
            self.X_reg_test,
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        # Train regression model
        self.reg_model = RandomForestRegressor(random_state=42)
        self.reg_model.fit(self.X_reg_train, self.y_reg_train)
    
    def _initialize_components(self):
        """Initialize DriftGuard components"""
        # Initialize drift detectors
        self.ks_detector = KSTestDriftDetector(self.config.drift)
        self.jsd_detector = JSDDriftDetector(self.config.drift)
        self.psi_detector = PSIDriftDetector(self.config.drift)
        
        # Initialize model monitors
        self.cls_monitor = ModelMonitor(self.config.monitor)
        self.reg_monitor = ModelMonitor(self.config.monitor)
        
        # Initialize data validator
        self.validator = DataValidator()
        
        # Initialize state manager
        self.state_manager = StateManager(
            path='tests/storage',
            retention_days=7
        )
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.config.alerts)
    
    def test_drift_detection(self):
        """Test drift detection methods"""
        # Initialize detectors with reference data
        self.ks_detector.initialize(self.cls_train_df)
        self.jsd_detector.initialize(self.cls_train_df)
        self.psi_detector.initialize(self.cls_train_df)
        
        # Test with normal data (no drift)
        normal_data = self.cls_test_df.copy()
        
        ks_reports = self.ks_detector.detect_drift(normal_data)
        self.assertTrue(all(not r.has_drift for r in ks_reports))
        
        jsd_reports = self.jsd_detector.detect_drift(normal_data)
        self.assertTrue(all(not r.has_drift for r in jsd_reports))
        
        psi_reports = self.psi_detector.detect_drift(normal_data)
        self.assertTrue(all(not r.has_drift for r in psi_reports))
        
        # Test with drifted data
        drift_data = normal_data.copy()
        drift_data['feature_0'] = drift_data['feature_0'] + 5.0
        
        ks_reports = self.ks_detector.detect_drift(drift_data)
        self.assertTrue(any(r.has_drift for r in ks_reports))
        
        jsd_reports = self.jsd_detector.detect_drift(drift_data)
        self.assertTrue(any(r.has_drift for r in jsd_reports))
        
        psi_reports = self.psi_detector.detect_drift(drift_data)
        self.assertTrue(any(r.has_drift for r in psi_reports))
    
    def test_model_monitoring(self):
        """Test model performance monitoring"""
        # Test classification monitoring
        self.cls_monitor.initialize("classification")
        
        cls_preds = pd.Series(
            self.cls_model.predict(self.X_cls_test)
        )
        cls_metrics = self.cls_monitor.track_performance(
            cls_preds,
            pd.Series(self.y_cls_test)
        )
        
        self.assertIn('accuracy', cls_metrics)
        self.assertIn('precision', cls_metrics)
        self.assertIn('recall', cls_metrics)
        self.assertIn('f1', cls_metrics)
        
        # Test regression monitoring
        self.reg_monitor.initialize("regression")
        
        reg_preds = pd.Series(
            self.reg_model.predict(self.X_reg_test)
        )
        reg_metrics = self.reg_monitor.track_performance(
            reg_preds,
            pd.Series(self.y_reg_test)
        )
        
        self.assertIn('mse', reg_metrics)
        self.assertIn('rmse', reg_metrics)
        self.assertIn('mae', reg_metrics)
        self.assertIn('r2', reg_metrics)
    
    def test_data_validation(self):
        """Test data validation"""
        # Initialize validator with reference data
        self.validator.initialize(self.cls_train_df)
        
        # Test with valid data
        valid_result = self.validator.validate(self.cls_test_df)
        self.assertTrue(valid_result.is_valid)
        self.assertEqual(len(valid_result.errors), 0)
        
        # Test with invalid data
        invalid_data = self.cls_test_df.copy()
        invalid_data['feature_0'] = 'invalid'
        
        invalid_result = self.validator.validate(invalid_data)
        self.assertFalse(invalid_result.is_valid)
        self.assertGreater(len(invalid_result.errors), 0)
    
    def test_state_management(self):
        """Test state management"""
        # Test state persistence
        test_state = {'test_key': 'test_value'}
        self.state_manager.save_state(test_state)
        
        loaded_state = self.state_manager.load_state()
        self.assertEqual(loaded_state['test_key'], 'test_value')
        
        # Test metrics tracking
        test_metrics = {
            'metric1': 0.95,
            'metric2': 0.85
        }
        self.state_manager.update_metrics(test_metrics)
        
        metrics_history = self.state_manager.get_metrics_history()
        self.assertGreater(len(metrics_history), 0)
    
    def test_alert_management(self):
        """Test alert management"""
        # Create test alert
        self.alert_manager.create_alert(
            message="Test alert",
            alert_type="test",
            severity="info",
            metadata={'test_key': 'test_value'}
        )
        
        # Get alerts
        alerts = self.alert_manager.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].message, "Test alert")
        
        # Clear alerts
        cleared = self.alert_manager.clear_alerts()
        self.assertEqual(cleared, 1)
        
        alerts = self.alert_manager.get_alerts()
        self.assertEqual(len(alerts), 0)

if __name__ == '__main__':
    unittest.main()
