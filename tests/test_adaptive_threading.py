"""Tests for adaptive thread pool sizing in drift detection."""
import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import patch, MagicMock
import logging

from driftguard.core.drift import (
    DriftDetector,
    DEFAULT_CPU_COUNT,
    WORKER_SCALE_FACTOR,
    MAX_WORKER_SCALE_FACTOR
)
from driftguard.core.config import DriftConfig


@pytest.fixture
def reference_data():
    """Generate reference data for testing"""
    np.random.seed(42)
    size = 1000
    
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, size),
        'feature2': np.random.uniform(0, 1, size),
        'feature3': np.random.exponential(2, size)
    })


@pytest.fixture
def test_data():
    """Generate test data for drift detection"""
    np.random.seed(43)
    size = 5000
    
    return pd.DataFrame({
        'feature1': np.random.normal(0.1, 1, size),
        'feature2': np.random.uniform(0, 1, size),
        'feature3': np.random.exponential(2, size)
    })


def test_adaptive_sizing_with_auto_scale(reference_data, test_data, caplog):
    """Test adaptive thread pool sizing with auto_scale_workers enabled"""
    # Create config with auto scaling enabled (default)
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    with caplog.at_level(logging.INFO):
        # Process data with multiple batches
        detector.detect(test_data, batch_size=1000)
    
    # Check that adaptive sizing was used and logged
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0, "Expected worker count to be logged"
    
    # Extract the number of workers from the log
    worker_count_str = worker_logs[0].split()[1]
    worker_count = int(worker_count_str)
    
    # Verify worker count is reasonable
    cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT
    num_batches = (len(test_data) + 999) // 1000  # 5 batches for 5000 rows
    expected_max = min(
        cpu_count * WORKER_SCALE_FACTOR,
        num_batches,
        cpu_count * MAX_WORKER_SCALE_FACTOR
    )
    
    assert worker_count <= expected_max
    assert worker_count > 0


def test_manual_worker_override(reference_data, test_data, caplog):
    """Test manual worker override via max_workers config"""
    # Set max_workers to a specific value
    max_workers_override = 3
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=max_workers_override
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    with caplog.at_level(logging.INFO):
        detector.detect(test_data, batch_size=1000)
    
    # Check that the manual override was used
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0
    
    # Extract and verify the worker count
    worker_count_str = worker_logs[0].split()[1]
    worker_count = int(worker_count_str)
    
    assert worker_count == max_workers_override


def test_auto_scale_disabled(reference_data, test_data, caplog):
    """Test that disabling auto_scale uses cpu_count"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=False,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    with caplog.at_level(logging.INFO):
        detector.detect(test_data, batch_size=1000)
    
    # Check that cpu_count was used
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0
    
    # Extract and verify the worker count
    worker_count_str = worker_logs[0].split()[1]
    worker_count = int(worker_count_str)
    
    cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT
    assert worker_count == cpu_count


def test_small_workload_uses_fewer_workers(reference_data, caplog):
    """Test that small workloads don't create unnecessary workers"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    # Small dataset that creates only 1 batch
    small_data = reference_data.iloc[:500]
    
    with caplog.at_level(logging.INFO):
        detector.detect(small_data, batch_size=1000)
    
    # Check worker count
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0
    
    # Extract the worker count
    worker_count_str = worker_logs[0].split()[1]
    worker_count = int(worker_count_str)
    
    # Should be 1 worker for 1 batch
    assert worker_count == 1


def test_worker_count_does_not_exceed_batches(reference_data, test_data, caplog):
    """Test that worker count doesn't exceed number of batches"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    # Create exactly 2 batches
    data_for_two_batches = test_data.iloc[:2000]
    
    with caplog.at_level(logging.INFO):
        detector.detect(data_for_two_batches, batch_size=1000)
    
    # Check worker count
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0
    
    # Extract the worker count
    worker_count_str = worker_logs[0].split()[1]
    worker_count = int(worker_count_str)
    
    # Should not exceed 2 workers for 2 batches
    assert worker_count <= 2


def test_zero_cpu_count_fallback(reference_data, test_data, caplog):
    """Test fallback when os.cpu_count() returns None"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    # Mock os.cpu_count to return None
    with patch('driftguard.core.drift.os.cpu_count', return_value=None):
        with caplog.at_level(logging.INFO):
            detector.detect(test_data, batch_size=1000)
        
        # Check that fallback value was used
        log_messages = [record.message for record in caplog.records]
        worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
        assert len(worker_logs) > 0
        
        # Extract the worker count
        worker_count_str = worker_logs[0].split()[1]
        worker_count = int(worker_count_str)
        
        # Should use fallback value
        assert worker_count > 0
        # Upper bound is DEFAULT_CPU_COUNT * MAX_WORKER_SCALE_FACTOR
        assert worker_count <= DEFAULT_CPU_COUNT * MAX_WORKER_SCALE_FACTOR


def test_config_defaults():
    """Test that DriftConfig has correct defaults for new fields"""
    config = DriftConfig()
    
    assert config.max_workers is None
    assert config.auto_scale_workers is True


def test_detect_method_returns_results(reference_data, test_data):
    """Test that detect method still returns valid results with new threading"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    results = detector.detect(test_data, batch_size=1000)
    
    # Verify results are returned
    assert isinstance(results, list)
    assert len(results) > 0


def test_empty_dataset_returns_empty_list(reference_data):
    """Test that empty input data returns empty list without error"""
    config = DriftConfig(
        methods=['ks'],
        auto_scale_workers=True,
        max_workers=None
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    # Empty dataframe
    empty_data = reference_data.iloc[0:0]
    results = detector.detect(empty_data, batch_size=1000)
    
    # Should return empty list, not raise error
    assert isinstance(results, list)
    assert len(results) == 0


def test_max_workers_validation():
    """Test that max_workers validation works correctly"""
    import pytest
    
    # Valid: None
    config1 = DriftConfig(max_workers=None)
    assert config1.max_workers is None
    
    # Valid: positive integer
    config2 = DriftConfig(max_workers=4)
    assert config2.max_workers == 4
    
    # Invalid: zero
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        DriftConfig(max_workers=0)
    
    # Invalid: negative
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        DriftConfig(max_workers=-1)


def test_explicit_single_worker(reference_data, test_data, caplog):
    """Test explicit max_workers=1 behavior"""
    config = DriftConfig(
        methods=['ks'],
        max_workers=1
    )
    
    detector = DriftDetector(config)
    detector.initialize(reference_data)
    
    with caplog.at_level(logging.INFO):
        results = detector.detect(test_data, batch_size=1000)
    
    # Should use exactly 1 worker
    log_messages = [record.message for record in caplog.records]
    worker_logs = [msg for msg in log_messages if 'workers for drift detection' in msg]
    assert len(worker_logs) > 0
    
    # Verify single worker was used
    assert '1 workers' in worker_logs[0] or '1 worker' in worker_logs[0]
    
    # Results should still be valid
    assert isinstance(results, list)
    assert len(results) > 0
