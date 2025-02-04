import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
import pandas as pd
from driftguard.drift_detector import DriftDetector

@pytest.fixture
def reference_data():
    """Fixture to create reference (training) data."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),  
        "feature2": np.random.uniform(0, 1, 1000)  
    })

@pytest.fixture
def similar_data():
    """Fixture to create new data similar to the reference (no drift)."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.uniform(0, 1, 1000)
    })

@pytest.fixture
def drifted_data():
    """Fixture to create new data that has significant drift."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.normal(3, 1, 1000),  
        "feature2": np.random.uniform(2, 3, 1000)  
    })

def test_drift_detector_initialization(reference_data):
    """Test if the DriftDetector initializes correctly."""
    detector = DriftDetector(reference_data)
    assert detector.reference_data.equals(reference_data)

def test_no_drift_detection(reference_data, similar_data):
    """Test case where no drift should be detected."""
    detector = DriftDetector(reference_data)
    drift_report = detector.detect_drift(similar_data)

    for feature, result in drift_report.items():
        assert result["p_value"] > 0.05  
        assert result["drift_score"] < 0.95  

def test_drift_detection(reference_data, drifted_data):
    """Test case where drift is expected to be detected."""
    detector = DriftDetector(reference_data)
    drift_report = detector.detect_drift(drifted_data)

    for feature, result in drift_report.items():
        assert result["p_value"] < 0.05  
        assert result["drift_score"] > 0.95  

def test_feature_drift_detection():
    """Test the feature-wise drift detection method directly."""
    detector = DriftDetector(pd.DataFrame())

    ref_feature = np.random.normal(0, 1, 1000)
    new_feature = np.random.normal(5, 1, 1000)  

    result = detector._detect_feature_drift(ref_feature, new_feature)
    
    assert result["p_value"] < 0.05  
    assert result["drift_score"] > 0.95  

def test_edge_cases():
    """Test edge cases like empty data and very small datasets."""
    detector = DriftDetector(pd.DataFrame())

    empty_df = pd.DataFrame()
    small_df1 = pd.DataFrame({"feature1": np.random.normal(0, 1, 3)})
    small_df2 = pd.DataFrame({"feature1": np.random.normal(5, 1, 3)})

    assert detector.detect_drift(empty_df) == {}

    drift_report = detector.detect_drift(small_df2)
    assert "feature1" in drift_report
    assert "p_value" in drift_report["feature1"]
    assert "drift_score" in drift_report["feature1"]
