"""
Drift detection module with various statistical tests.
"""
from typing import List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig

class BaseDriftDetector(IDriftDetector):
    """Base class for drift detection"""
    
    def __init__(self, config: DriftConfig):
        """Initialize drift detector"""
        self.config = config
        self.reference_data = None
        self._initialized = False
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        self.reference_data = reference_data
        self._initialized = True
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        self.reference_data = new_reference

class KSTestDriftDetector(BaseDriftDetector):
    """Drift detector using Kolmogorov-Smirnov test"""
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using KS test"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if len(new_data) < self.config.min_samples:
            raise ValueError(
                f"Not enough samples. Got {len(new_data)}, "
                f"need at least {self.config.min_samples}"
            )
        
        reports = []
        for column in self.reference_data.columns:
            ref_values = self.reference_data[column].dropna().values
            new_values = new_data[column].dropna().values
            
            if len(ref_values) == 0 or len(new_values) == 0:
                continue
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(ref_values, new_values)
            
            reports.append(
                DriftReport(
                    feature_name=column,
                    drift_score=statistic,
                    p_value=p_value,
                    threshold=self.config.threshold,
                    method="ks_test",
                    has_drift=p_value < self.config.threshold
                )
            )
        
        return reports

class JSDDriftDetector(BaseDriftDetector):
    """Drift detector using Jensen-Shannon Divergence"""
    
    def _compute_histogram(self, data: np.ndarray, bins: int = 20) -> np.ndarray:
        """Compute normalized histogram"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist / np.sum(hist)  # Ensure normalization
        return hist
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using JSD"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if len(new_data) < self.config.min_samples:
            raise ValueError(
                f"Not enough samples. Got {len(new_data)}, "
                f"need at least {self.config.min_samples}"
            )
        
        reports = []
        for column in self.reference_data.columns:
            ref_values = self.reference_data[column].dropna().values
            new_values = new_data[column].dropna().values
            
            if len(ref_values) == 0 or len(new_values) == 0:
                continue
            
            # Compute histograms
            ref_hist = self._compute_histogram(ref_values)
            new_hist = self._compute_histogram(new_values)
            
            # Compute JSD
            jsd = jensenshannon(ref_hist, new_hist)
            
            reports.append(
                DriftReport(
                    feature_name=column,
                    drift_score=float(jsd),
                    threshold=self.config.threshold,
                    method="jsd",
                    has_drift=jsd > self.config.threshold
                )
            )
        
        return reports

class PSIDriftDetector(BaseDriftDetector):
    """Drift detector using Population Stability Index"""
    
    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Compute Population Stability Index"""
        # Create bins based on reference data
        bin_edges = np.percentile(
            reference,
            np.linspace(0, 100, bins + 1)
        )
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages with small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_pct = ref_hist / len(reference) + epsilon
        cur_pct = cur_hist / len(current) + epsilon
        
        # Calculate PSI
        psi = np.sum(
            (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        )
        
        return float(psi)
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using PSI"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if len(new_data) < self.config.min_samples:
            raise ValueError(
                f"Not enough samples. Got {len(new_data)}, "
                f"need at least {self.config.min_samples}"
            )
        
        reports = []
        for column in self.reference_data.columns:
            ref_values = self.reference_data[column].dropna().values
            new_values = new_data[column].dropna().values
            
            if len(ref_values) == 0 or len(new_values) == 0:
                continue
            
            # Compute PSI
            psi = self._compute_psi(ref_values, new_values)
            
            reports.append(
                DriftReport(
                    feature_name=column,
                    drift_score=psi,
                    threshold=self.config.threshold,
                    method="psi",
                    has_drift=psi > self.config.threshold
                )
            )
        
        return reports

def create_drift_detector(
    method: DriftMethod,
    config: DriftConfig
) -> IDriftDetector:
    """Create drift detector instance"""
    detectors: Dict[DriftMethod, Type[BaseDriftDetector]] = {
        DriftMethod.KS_TEST: KSTestDriftDetector,
        DriftMethod.JSD: JSDDriftDetector,
        DriftMethod.PSI: PSIDriftDetector
    }
    
    detector_class = detectors.get(method)
    if detector_class is None:
        raise ValueError(f"Unknown drift detection method: {method}")
    
    return detector_class(config)
