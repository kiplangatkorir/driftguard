"""
Test suite for batch feature processing functionality.
"""
import unittest
import numpy as np
import pandas as pd

from driftguard.core.config import DriftConfig
from driftguard.core.drift import DriftDetector


class TestBatchFeatureProcessing(unittest.TestCase):
    """Test batch feature processing functionality"""
    
    def setUp(self):
        """Setup test environment for batch processing"""
        # Create test data with continuous and categorical features
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'cont_1': np.random.normal(0, 1, 500),
            'cont_2': np.random.normal(0, 1, 500),
            'cont_3': np.random.normal(0, 1, 500),
            'cat_1': pd.Categorical(np.random.choice(['A', 'B', 'C'], 500)),
            'cat_2': pd.Categorical(np.random.choice(['X', 'Y', 'Z'], 500)),
        })
        
        self.test_data = pd.DataFrame({
            'cont_1': np.random.normal(0.1, 1, 500),
            'cont_2': np.random.normal(0.1, 1, 500),
            'cont_3': np.random.normal(0.1, 1, 500),
            'cat_1': pd.Categorical(np.random.choice(['A', 'B', 'C'], 500)),
            'cat_2': pd.Categorical(np.random.choice(['X', 'Y', 'Z'], 500)),
        })
        
        # Initialize detectors
        self.batch_config = DriftConfig(
            methods=['ks', 'jsd', 'psi'],
            batch_feature_processing=True
        )
        self.individual_config = DriftConfig(
            methods=['ks', 'jsd', 'psi'],
            batch_feature_processing=False
        )
    
    def test_feature_grouping(self):
        """Test that features are correctly grouped by type"""
        detector = DriftDetector(self.batch_config)
        detector.initialize(self.reference_data)
        
        feature_groups = detector._group_features_by_type()
        
        # Check that continuous features are grouped correctly
        self.assertIn('continuous', feature_groups)
        self.assertIn('categorical', feature_groups)
        self.assertEqual(len(feature_groups['continuous']), 3)
        self.assertEqual(len(feature_groups['categorical']), 2)
        self.assertIn('cont_1', feature_groups['continuous'])
        self.assertIn('cont_2', feature_groups['continuous'])
        self.assertIn('cont_3', feature_groups['continuous'])
        self.assertIn('cat_1', feature_groups['categorical'])
        self.assertIn('cat_2', feature_groups['categorical'])
    
    def test_batch_vs_individual_processing(self):
        """Test that batch processing produces same results as individual processing"""
        # Initialize both detectors
        batch_detector = DriftDetector(self.batch_config)
        batch_detector.initialize(self.reference_data)
        
        individual_detector = DriftDetector(self.individual_config)
        individual_detector.initialize(self.reference_data)
        
        # Run detection with both methods
        batch_reports = batch_detector.detect(self.test_data)
        individual_reports = individual_detector.detect(self.test_data)
        
        # Check that number of reports is the same
        self.assertEqual(len(batch_reports), len(individual_reports))
        
        # Sort reports by feature and method for comparison
        batch_sorted = sorted(batch_reports, key=lambda r: (r.features[0], r.method))
        individual_sorted = sorted(individual_reports, key=lambda r: (r.features[0], r.method))
        
        # Compare each report
        for batch_report, individual_report in zip(batch_sorted, individual_sorted):
            self.assertEqual(batch_report.method, individual_report.method)
            self.assertEqual(batch_report.features, individual_report.features)
            # Scores should be very close (accounting for floating point precision)
            self.assertAlmostEqual(batch_report.score, individual_report.score, places=10)
            self.assertEqual(batch_report.has_drift, individual_report.has_drift)
    
    def test_batch_processing_with_large_dataset(self):
        """Test performance with large feature sets"""
        # Create larger dataset with more features
        np.random.seed(42)
        large_reference = pd.DataFrame({
            f'cont_{i}': np.random.normal(0, 1, 1000) for i in range(20)
        })
        large_reference.update({
            f'cat_{i}': pd.Categorical(np.random.choice(['A', 'B', 'C'], 1000)) 
            for i in range(10)
        })
        
        large_test = pd.DataFrame({
            f'cont_{i}': np.random.normal(0.1, 1, 1000) for i in range(20)
        })
        large_test.update({
            f'cat_{i}': pd.Categorical(np.random.choice(['A', 'B', 'C'], 1000)) 
            for i in range(10)
        })
        
        # Test with batch processing
        detector = DriftDetector(self.batch_config)
        detector.initialize(large_reference)
        
        # This should complete without errors
        reports = detector.detect(large_test)
        
        # Verify reports were generated for all features and methods
        expected_reports = len(self.batch_config.methods) * len(large_reference.columns)
        # Some methods may return None for incompatible feature types
        self.assertGreater(len(reports), 0)
        self.assertLessEqual(len(reports), expected_reports)
    
    def test_process_features_batch(self):
        """Test _process_features_batch method directly"""
        detector = DriftDetector(self.batch_config)
        detector.initialize(self.reference_data)
        
        # Test processing continuous features
        continuous_features = ['cont_1', 'cont_2', 'cont_3']
        reports = detector._process_features_batch(
            self.test_data, continuous_features, 'ks'
        )
        
        # Should have reports for all continuous features
        self.assertEqual(len(reports), len(continuous_features))
        for report in reports:
            self.assertEqual(report.method, 'ks')
            self.assertIn(report.features[0], continuous_features)


if __name__ == '__main__':
    unittest.main()
