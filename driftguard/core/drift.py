"""
Drift detection module for DriftGuard.
Implements multiple statistical methods for drift detection:
- Kolmogorov-Smirnov (KS) test
- Jensen-Shannon Divergence (JSD)
- Population Stability Index (PSI)
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr
from datetime import datetime
import logging
from .interfaces import IDriftDetector, DriftReport, DriftConfig

logger = logging.getLogger(__name__)

class BaseDriftDetector(IDriftDetector):
    """Base class for drift detection"""
    
    def __init__(self, config: DriftConfig):
        """Initialize drift detector"""
        self.config = config
        self.reference_data = None
        self.feature_names = None
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        self.reference_data = reference_data
        self.feature_names = list(reference_data.columns)
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        if set(new_reference.columns) != set(self.feature_names):
            raise ValueError("New reference data must have same features")
        self.reference_data = new_reference
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        raise NotImplementedError

class KSTestDriftDetector(BaseDriftDetector):
    """Drift detector using Kolmogorov-Smirnov test"""
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using KS test"""
        if not self.reference_data is not None:
            raise ValueError("Reference data not initialized")
            
        reports = []
        for feature in self.feature_names:
            # Get feature values
            ref_values = self.reference_data[feature].values
            new_values = new_data[feature].values
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(ref_values, new_values)
            
            reports.append(DriftReport(
                feature_name=feature,
                drift_score=1 - p_value,  # Convert p-value to drift score
                p_value=p_value,
                test_statistic=statistic,
                threshold=self.config.threshold
            ))
        
        return reports

class JSDDriftDetector(BaseDriftDetector):
    """Drift detector using Jensen-Shannon Divergence"""
    
    def _calculate_jsd(
        self,
        p: np.ndarray,
        q: np.ndarray,
        bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon Divergence"""
        # Create histograms
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
        
        # Add small constant to avoid zero probabilities
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # Calculate JSD
        m = 0.5 * (p_hist + q_hist)
        jsd = 0.5 * (
            np.sum(rel_entr(p_hist, m)) +
            np.sum(rel_entr(q_hist, m))
        )
        
        return jsd
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using JSD"""
        if not self.reference_data is not None:
            raise ValueError("Reference data not initialized")
            
        reports = []
        for feature in self.feature_names:
            # Get feature values
            ref_values = self.reference_data[feature].values
            new_values = new_data[feature].values
            
            # Calculate JSD
            jsd = self._calculate_jsd(ref_values, new_values)
            
            reports.append(DriftReport(
                feature_name=feature,
                drift_score=jsd,
                threshold=self.config.threshold
            ))
        
        return reports

class PSIDriftDetector(BaseDriftDetector):
    """Drift detector using Population Stability Index"""
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on expected distribution
        bin_edges = np.percentile(
            expected,
            np.linspace(0, 100, bins + 1)
        )
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')
        
        # Calculate distributions
        expected_counts = np.histogram(expected, bins=bin_edges)[0] + 1e-10
        actual_counts = np.histogram(actual, bins=bin_edges)[0] + 1e-10
        
        # Convert to percentages
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Calculate PSI
        psi = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )
        
        return psi
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using PSI"""
        if not self.reference_data is not None:
            raise ValueError("Reference data not initialized")
            
        reports = []
        for feature in self.feature_names:
            # Get feature values
            ref_values = self.reference_data[feature].values
            new_values = new_data[feature].values
            
            # Calculate PSI
            psi = self._calculate_psi(ref_values, new_values)
            
            reports.append(DriftReport(
                feature_name=feature,
                drift_score=psi,
                threshold=self.config.threshold
            ))
        
        return reports

def create_drift_detector(
    method: str,
    config: DriftConfig
) -> BaseDriftDetector:
    """Factory function to create drift detector"""
    detectors = {
        'ks_test': KSTestDriftDetector,
        'jsd': JSDDriftDetector,
        'psi': PSIDriftDetector
    }
    
    if method not in detectors:
        raise ValueError(
            f"Unknown drift detection method: {method}. "
            f"Available methods: {list(detectors.keys())}"
        )
    
    return detectors[method](config)
