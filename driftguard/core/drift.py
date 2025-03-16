"""
Drift detection module for DriftGuard.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from datetime import datetime

from .interfaces import IDriftDetector, DriftReport, DriftMethod
from .config import DriftConfig

class BaseDriftDetector(IDriftDetector):
    """Base class for drift detectors"""
    
    def __init__(self, config: DriftConfig):
        """Initialize drift detector"""
        self.config = config
        self.reference_data = None
        self._initialized = False
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        self.reference_data = reference_data.copy()
        self._initialized = True
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if not all(col in data.columns for col in self.reference_data.columns):
            raise ValueError("Missing columns in input data")

class KSTestDriftDetector(BaseDriftDetector):
    """Kolmogorov-Smirnov test for drift detection"""
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using KS test"""
        self._validate_data(new_data)
        
        reports = []
        n_features = len(self.reference_data.columns)
        alpha = self.config.ks_test["threshold"]
        
        # Apply Bonferroni correction if specified
        if self.config.ks_test["correction"] == "bonferroni":
            alpha = alpha / n_features
        
        for feature in self.reference_data.columns:
            ref_data = self.reference_data[feature].dropna()
            new_data_feature = new_data[feature].dropna()
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                ref_data,
                new_data_feature
            )
            
            reports.append(
                DriftReport(
                    feature=feature,
                    method=DriftMethod.KS_TEST,
                    statistic=float(statistic),
                    p_value=float(p_value),
                    threshold=alpha,
                    has_drift=p_value < alpha,
                    timestamp=datetime.now()
                )
            )
        
        return reports

class JSDDriftDetector(BaseDriftDetector):
    """Jensen-Shannon Divergence for drift detection"""
    
    def _compute_histogram(
        self,
        data: pd.Series,
        bins: int
    ) -> np.ndarray:
        """Compute normalized histogram"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist / np.sum(hist)  # Normalize
        return hist
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using JSD"""
        self._validate_data(new_data)
        
        reports = []
        threshold = self.config.jsd["threshold"]
        bins = self.config.jsd["bins"]
        
        for feature in self.reference_data.columns:
            ref_data = self.reference_data[feature].dropna()
            new_data_feature = new_data[feature].dropna()
            
            # Compute histograms
            ref_hist = self._compute_histogram(ref_data, bins)
            new_hist = self._compute_histogram(new_data_feature, bins)
            
            # Compute JSD
            jsd = jensenshannon(ref_hist, new_hist)
            
            reports.append(
                DriftReport(
                    feature=feature,
                    method=DriftMethod.JSD,
                    statistic=float(jsd),
                    threshold=threshold,
                    has_drift=jsd > threshold,
                    timestamp=datetime.now()
                )
            )
        
        return reports

class PSIDriftDetector(BaseDriftDetector):
    """Population Stability Index for drift detection"""
    
    def _compute_psi(
        self,
        ref_data: pd.Series,
        new_data: pd.Series,
        bins: int
    ) -> float:
        """Compute PSI"""
        # Compute bin edges using reference data
        bin_edges = np.percentile(
            ref_data,
            np.linspace(0, 100, bins + 1)
        )
        
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        
        # Compute histograms
        ref_hist, _ = np.histogram(ref_data, bins=bin_edges)
        new_hist, _ = np.histogram(new_data, bins=bin_edges)
        
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
        self._validate_data(new_data)
        
        reports = []
        threshold = self.config.psi["threshold"]
        bins = self.config.psi["bins"]
        
        for feature in self.reference_data.columns:
            ref_data = self.reference_data[feature].dropna()
            new_data_feature = new_data[feature].dropna()
            
            # Skip if not enough unique values
            if len(np.unique(ref_data)) < 2:
                continue
            
            # Compute PSI
            psi = self._compute_psi(ref_data, new_data_feature, bins)
            
            reports.append(
                DriftReport(
                    feature=feature,
                    method=DriftMethod.PSI,
                    statistic=psi,
                    threshold=threshold,
                    has_drift=psi > threshold,
                    timestamp=datetime.now()
                )
            )
        
        return reports
