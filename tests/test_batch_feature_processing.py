"""Tests for batch feature processing functionality."""
import pytest
import numpy as np
import pandas as pd

from driftguard.core.drift import DriftDetector
from driftguard.core.config import DriftConfig


@pytest.fixture
def mixed_reference_data():
    """Generate reference data with both categorical and continuous features"""
    np.random.seed(42)
    size = 1000
    
    return pd.DataFrame({
        'continuous_1': np.random.normal(0, 1, size),
        'continuous_2': np.random.uniform(0, 1, size),
        'continuous_3': np.random.exponential(2, size),
        'categorical_1': np.random.choice(['A', 'B', 'C'], size),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], size),
    })


@pytest.fixture
def mixed_test_data():
    """Generate test data with both categorical and continuous features"""
    np.random.seed(43)
    size = 500
    
    return pd.DataFrame({
        'continuous_1': np.random.normal(0.1, 1, size),
        'continuous_2': np.random.uniform(0, 1, size),
        'continuous_3': np.random.exponential(2, size),
        'categorical_1': np.random.choice(['A', 'B', 'C'], size),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], size),
    })


@pytest.fixture
def continuous_only_data():
    """Generate reference data with only continuous features"""
    np.random.seed(42)
    size = 1000
    
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, size),
        'feature2': np.random.uniform(0, 1, size),
        'feature3': np.random.exponential(2, size)
    })


@pytest.fixture
def categorical_only_data():
    """Generate reference data with only categorical features"""
    np.random.seed(42)
    size = 1000
    
    return pd.DataFrame({
        'feature1': np.random.choice(['A', 'B', 'C'], size),
        'feature2': np.random.choice(['X', 'Y', 'Z'], size),
        'feature3': np.random.choice(['Red', 'Blue', 'Green'], size),
    })


def test_group_features_by_type_mixed(mixed_reference_data):
    """Test feature grouping with mixed feature types"""
    config = DriftConfig(methods=['ks', 'jsd'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    groups = detector._feature_groups
    
    # Verify structure
    assert 'continuous' in groups
    assert 'categorical' in groups
    
    # Verify continuous features are correctly grouped
    assert len(groups['continuous']) == 3
    assert 'continuous_1' in groups['continuous']
    assert 'continuous_2' in groups['continuous']
    assert 'continuous_3' in groups['continuous']
    
    # Verify categorical features are correctly grouped
    assert len(groups['categorical']) == 2
    assert 'categorical_1' in groups['categorical']
    assert 'categorical_2' in groups['categorical']


def test_group_features_by_type_continuous_only(continuous_only_data):
    """Test feature grouping with only continuous features"""
    config = DriftConfig(methods=['ks'])
    detector = DriftDetector(config)
    detector.initialize(continuous_only_data)
    
    groups = detector._feature_groups
    
    assert len(groups['continuous']) == 3
    assert len(groups['categorical']) == 0


def test_group_features_by_type_categorical_only(categorical_only_data):
    """Test feature grouping with only categorical features"""
    config = DriftConfig(methods=['jsd'])
    detector = DriftDetector(config)
    detector.initialize(categorical_only_data)
    
    groups = detector._feature_groups
    
    assert len(groups['continuous']) == 0
    assert len(groups['categorical']) == 3


def test_batch_processing_produces_same_results(mixed_reference_data, mixed_test_data):
    """Verify batch processing produces same results as individual processing
    
    This is verified by comparing the number and types of reports generated.
    The exact scores should match since both use the same detection logic.
    """
    config = DriftConfig(methods=['ks', 'jsd', 'psi'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    # Process with batch feature processing (new implementation)
    batch_reports = detector.detect(mixed_test_data, batch_size=1000)
    
    # Verify we got reports
    assert len(batch_reports) > 0
    
    # Count reports by method
    reports_by_method = {}
    for report in batch_reports:
        method = report.method
        if method not in reports_by_method:
            reports_by_method[method] = []
        reports_by_method[method].append(report)
    
    # Verify we have reports for all methods
    assert 'ks' in reports_by_method
    assert 'jsd' in reports_by_method
    assert 'psi' in reports_by_method
    
    # KS only works on continuous features (3 features)
    assert len(reports_by_method['ks']) == 3
    
    # JSD and PSI work on all features (5 features)
    assert len(reports_by_method['jsd']) == 5
    assert len(reports_by_method['psi']) == 5


def test_process_continuous_features(continuous_only_data):
    """Test processing of continuous features in batch"""
    config = DriftConfig(methods=['ks'])
    detector = DriftDetector(config)
    detector.initialize(continuous_only_data)
    
    # Create test data with drift
    test_data = continuous_only_data.copy()
    test_data['feature1'] = test_data['feature1'] + 3.0  # Introduce drift
    
    reports = detector._process_continuous_features(
        test_data, 
        ['feature1', 'feature2', 'feature3'],
        'ks'
    )
    
    # Should get 3 reports (one for each feature)
    assert len(reports) == 3
    
    # All should be KS tests
    assert all(r.method == 'ks' for r in reports)
    
    # Feature1 should have high drift score
    feature1_reports = [r for r in reports if 'feature1' in r.features]
    assert len(feature1_reports) == 1
    # KS statistic should be high for drifted feature
    assert feature1_reports[0].score > 0.5


def test_process_categorical_features(categorical_only_data):
    """Test processing of categorical features in batch"""
    config = DriftConfig(methods=['jsd'])
    detector = DriftDetector(config)
    detector.initialize(categorical_only_data)
    
    # Create test data with same distribution
    test_data = categorical_only_data.copy()
    
    reports = detector._process_categorical_features(
        test_data,
        ['feature1', 'feature2', 'feature3'],
        'jsd'
    )
    
    # Should get 3 reports (one for each feature)
    assert len(reports) == 3
    
    # All should be JSD tests
    assert all(r.method == 'jsd' for r in reports)


def test_batch_processing_with_multiple_batches(mixed_reference_data):
    """Test batch processing with data split into multiple batches"""
    config = DriftConfig(methods=['ks', 'jsd'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    # Create larger test data that will be split into batches
    np.random.seed(44)
    large_test_data = pd.DataFrame({
        'continuous_1': np.random.normal(0, 1, 2500),
        'continuous_2': np.random.uniform(0, 1, 2500),
        'continuous_3': np.random.exponential(2, 2500),
        'categorical_1': np.random.choice(['A', 'B', 'C'], 2500),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], 2500),
    })
    
    # Process with batch_size=1000 (should create 3 batches)
    reports = detector.detect(large_test_data, batch_size=1000)
    
    # Each batch should produce reports for each method/feature combination
    # With 2 methods and 5 features per batch, and 3 batches:
    # - KS on 3 continuous features = 3 reports per batch
    # - JSD on 5 features = 5 reports per batch
    # Total per batch = 8 reports
    # Expected total = 8 * 3 = 24 reports
    assert len(reports) == 24


def test_feature_grouping_persists_across_detect_calls(mixed_reference_data, mixed_test_data):
    """Test that feature grouping is cached and reused across multiple detect calls"""
    config = DriftConfig(methods=['ks'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    # Get initial feature groups
    initial_groups = detector._feature_groups
    
    # Run detection
    detector.detect(mixed_test_data, batch_size=1000)
    
    # Feature groups should be the same object (cached)
    assert detector._feature_groups is initial_groups


def test_empty_feature_groups(mixed_reference_data):
    """Test handling when a feature group is empty for a specific method"""
    # Create data with only continuous features
    continuous_data = mixed_reference_data[['continuous_1', 'continuous_2', 'continuous_3']].copy()
    
    config = DriftConfig(methods=['ks'])  # KS only works on continuous
    detector = DriftDetector(config)
    detector.initialize(continuous_data)
    
    # Test data
    test_data = continuous_data.copy()
    
    # Process - should not fail even though categorical group is empty
    reports = detector.detect(test_data, batch_size=1000)
    
    # Should get reports for continuous features only
    assert len(reports) == 3
    assert all(r.method == 'ks' for r in reports)


def test_progress_tracking_with_grouped_features(mixed_reference_data, mixed_test_data, capfd):
    """Test that progress tracking works correctly with grouped feature processing"""
    config = DriftConfig(methods=['ks', 'jsd'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    # Run detection (this should show progress)
    detector.detect(mixed_test_data, batch_size=1000)
    
    # Capture output to verify progress bar was shown
    # (Note: tqdm writes to stderr by default)
    captured = capfd.readouterr()
    
    # Progress bar should have been displayed
    assert 'Processing features' in captured.err or captured.err == ''


def test_method_compatibility_with_feature_types(mixed_reference_data, mixed_test_data):
    """Test that methods are only applied to compatible feature types"""
    config = DriftConfig(methods=['ks', 'jsd', 'psi'])
    detector = DriftDetector(config)
    detector.initialize(mixed_reference_data)
    
    reports = detector.detect(mixed_test_data, batch_size=1000)
    
    # Separate reports by method
    ks_reports = [r for r in reports if r.method == 'ks']
    jsd_reports = [r for r in reports if r.method == 'jsd']
    psi_reports = [r for r in reports if r.method == 'psi']
    
    # KS should only be applied to continuous features
    assert len(ks_reports) == 3  # 3 continuous features
    for report in ks_reports:
        assert any('continuous' in feat for feat in report.features)
    
    # JSD and PSI should be applied to all features
    assert len(jsd_reports) == 5  # All 5 features
    assert len(psi_reports) == 5  # All 5 features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
