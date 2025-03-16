"""
Drift detection module implementing various statistical tests.
"""
from typing import Dict, List, Optional, Type
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from datetime import datetime
import logging
from .interfaces import IDriftDetector, DriftReport
from .config import DriftMethod, DriftConfig

logger = logging.getLogger(__name__)

class BaseDriftDetector(IDriftDetector):
    """Base class for drift detection"""
    
    def __init__(self, config: DriftConfig):
        """Initialize detector"""
        self.config = config
        self.reference_data = None
        self.feature_stats = {}
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        try:
            self.reference_data = reference_data.copy()
            self._compute_feature_stats()
            logger.info("Initialized drift detector")
        except Exception as e:
            logger.error(f"Failed to initialize drift detector: {str(e)}")
            raise
    
    def _compute_feature_stats(self) -> None:
        """Compute feature statistics"""
        for col in self.reference_data.columns:
            self.feature_stats[col] = {
                'mean': float(self.reference_data[col].mean()),
                'std': float(self.reference_data[col].std()),
                'min': float(self.reference_data[col].min()),
                'max': float(self.reference_data[col].max())
            }
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        raise NotImplementedError
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        try:
            self.reference_data = new_reference.copy()
            self._compute_feature_stats()
            logger.info("Updated reference data")
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")
            raise

class KSTestDriftDetector(BaseDriftDetector):
    """Drift detector using Kolmogorov-Smirnov test"""
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using KS test"""
        if self.reference_data is None:
            raise ValueError("Detector not initialized")
        
        reports = []
        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue
            
            try:
                # Perform KS test
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col],
                    new_data[col]
                )
                
                reports.append(DriftReport(
                    feature_name=col,
                    drift_score=statistic,
                    p_value=p_value,
                    threshold=self.config.threshold,
                    method="ks_test",
                    has_drift=p_value < self.config.threshold
                ))
                
            except Exception as e:
                logger.error(
                    f"Failed to compute KS test for feature {col}: {str(e)}"
                )
        
        return reports

class JSDDriftDetector(BaseDriftDetector):
    """Drift detector using Jensen-Shannon Divergence"""
    
    def _compute_histogram(self, data: pd.Series, bins: int = 50) -> np.ndarray:
        """Compute normalized histogram"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist / np.sum(hist)
        return hist
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using JSD"""
        if self.reference_data is None:
            raise ValueError("Detector not initialized")
        
        reports = []
        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue
            
            try:
                # Compute histograms
                ref_hist = self._compute_histogram(self.reference_data[col])
                new_hist = self._compute_histogram(new_data[col])
                
                # Compute JSD
                jsd = jensenshannon(ref_hist, new_hist)
                
                reports.append(DriftReport(
                    feature_name=col,
                    drift_score=float(jsd),
                    threshold=self.config.threshold,
                    method="jensen_shannon",
                    has_drift=jsd > self.config.threshold
                ))
                
            except Exception as e:
                logger.error(
                    f"Failed to compute JSD for feature {col}: {str(e)}"
                )
        
        return reports

class PSIDriftDetector(BaseDriftDetector):
    """Drift detector using Population Stability Index"""
    
    def _compute_psi(
        self,
        reference: pd.Series,
        new: pd.Series,
        bins: int = 10
    ) -> float:
        """Compute Population Stability Index"""
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        new_hist, _ = np.histogram(new, bins=bin_edges)
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        new_hist = new_hist + epsilon
        
        # Normalize histograms
        ref_hist = ref_hist / np.sum(ref_hist)
        new_hist = new_hist / np.sum(new_hist)
        
        # Compute PSI
        psi = np.sum(
            (new_hist - ref_hist) * np.log(new_hist / ref_hist)
        )
        
        return float(psi)
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using PSI"""
        if self.reference_data is None:
            raise ValueError("Detector not initialized")
        
        reports = []
        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue
            
            try:
                # Compute PSI
                psi = self._compute_psi(
                    self.reference_data[col],
                    new_data[col]
                )
                
                reports.append(DriftReport(
                    feature_name=col,
                    drift_score=psi,
                    threshold=self.config.threshold,
                    method="psi",
                    has_drift=psi > self.config.threshold
                ))
                
            except Exception as e:
                logger.error(
                    f"Failed to compute PSI for feature {col}: {str(e)}"
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
