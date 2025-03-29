"""
Drift detection module for DriftGuard.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from scipy.special import rel_entr
from collections import defaultdict
import logging

from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig

logger = logging.getLogger(__name__)

class DriftDetector(IDriftDetector):
    """Detects data drift using multiple statistical methods"""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """Initialize drift detector"""
        self.config = config or DriftConfig()
        self.reference_data = None
        self.feature_types = {}
        self.reference_stats = {}
        self._initialized = False
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize detector with reference data"""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
        
        self.reference_data = reference_data.copy()
        
        # Determine feature types
        self.feature_types = {
            col: 'categorical' if pd.api.types.is_categorical_dtype(reference_data[col])
            or pd.api.types.is_object_dtype(reference_data[col])
            else 'continuous'
            for col in reference_data.columns
        }
        
        # Compute reference statistics
        self.reference_stats = {}
        for col in reference_data.columns:
            if self.feature_types[col] == 'continuous':
                values = reference_data[col].dropna()
                self.reference_stats[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'hist': np.histogram(values, bins=20)[0]
                }
            else:
                self.reference_stats[col] = {
                    'value_counts': reference_data[col].value_counts(normalize=True)
                }
        
        self._initialized = True
    
    def detect(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        reports = []
        
        # Detect drift for each feature using configured methods
        for method in self.config.methods:
            if method == 'ks':
                reports.extend(self._detect_ks(data))
            elif method == 'jsd':
                reports.extend(self._detect_jsd(data))
            elif method == 'psi':
                reports.extend(self._detect_psi(data))
            elif method == 'wasserstein':
                reports.extend(self._detect_wasserstein(data))
            elif method == 'chi2':
                reports.extend(self._detect_chi2(data))
        
        return reports
    
    def _detect_ks(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Kolmogorov-Smirnov test"""
        reports = []
        threshold = self.config.thresholds['ks']
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'continuous':
                continue
            
            ref_values = self.reference_data[col].dropna()
            new_values = data[col].dropna()
            
            if len(new_values) < 2:
                continue
            
            statistic, _ = stats.ks_2samp(ref_values, new_values)
            
            reports.append(DriftReport(
                method='ks',
                score=statistic,
                threshold=threshold,
                features=[col]
            ))
        
        return reports
    
    def _detect_jsd(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Jensen-Shannon divergence"""
        reports = []
        threshold = self.config.thresholds['jsd']
        
        for col in self.reference_data.columns:
            ref_hist = None
            new_hist = None
            
            if self.feature_types[col] == 'continuous':
                ref_hist = self.reference_stats[col]['hist']
                new_hist = np.histogram(
                    data[col].dropna(),
                    bins=20
                )[0]
            else:
                ref_counts = self.reference_stats[col]['value_counts']
                new_counts = data[col].value_counts(normalize=True)
                
                # Align categories
                all_categories = sorted(
                    set(ref_counts.index) | set(new_counts.index)
                )
                ref_hist = np.array([
                    ref_counts.get(cat, 0) for cat in all_categories
                ])
                new_hist = np.array([
                    new_counts.get(cat, 0) for cat in all_categories
                ])
            
            # Add smoothing to avoid division by zero
            ref_hist = ref_hist + 1e-10
            new_hist = new_hist + 1e-10
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            new_hist = new_hist / new_hist.sum()
            
            # Calculate JSD
            m = 0.5 * (ref_hist + new_hist)
            jsd = 0.5 * (
                sum(rel_entr(ref_hist, m)) +
                sum(rel_entr(new_hist, m))
            )
            
            reports.append(DriftReport(
                method='jsd',
                score=jsd,
                threshold=threshold,
                features=[col]
            ))
        
        return reports
    
    def _detect_psi(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Population Stability Index"""
        reports = []
        threshold = self.config.thresholds['psi']
        
        for col in self.reference_data.columns:
            ref_hist = None
            new_hist = None
            
            if self.feature_types[col] == 'continuous':
                ref_hist = self.reference_stats[col]['hist']
                new_hist = np.histogram(
                    data[col].dropna(),
                    bins=20
                )[0]
            else:
                ref_counts = self.reference_stats[col]['value_counts']
                new_counts = data[col].value_counts(normalize=True)
                
                # Align categories
                all_categories = sorted(
                    set(ref_counts.index) | set(new_counts.index)
                )
                ref_hist = np.array([
                    ref_counts.get(cat, 0) for cat in all_categories
                ])
                new_hist = np.array([
                    new_counts.get(cat, 0) for cat in all_categories
                ])
            
            # Add smoothing to avoid division by zero
            ref_hist = ref_hist + 1e-10
            new_hist = new_hist + 1e-10
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            new_hist = new_hist / new_hist.sum()
            
            # Calculate PSI
            psi = sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))
            
            reports.append(DriftReport(
                method='psi',
                score=psi,
                threshold=threshold,
                features=[col]
            ))
        
        return reports
    
    def _detect_wasserstein(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Wasserstein distance"""
        reports = []
        threshold = self.config.thresholds.get('wasserstein', 0.1)
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'continuous':
                continue
            
            ref_values = self.reference_data[col].dropna()
            new_values = data[col].dropna()
            
            if len(new_values) < 2:
                continue
            
            # Normalize values to [0,1] range
            ref_min = ref_values.min()
            ref_max = ref_values.max()
            if ref_max > ref_min:
                ref_norm = (ref_values - ref_min) / (ref_max - ref_min)
                new_norm = (new_values - ref_min) / (ref_max - ref_min)
                
                distance = wasserstein_distance(ref_norm, new_norm)
                
                reports.append(DriftReport(
                    method='wasserstein',
                    score=distance,
                    threshold=threshold,
                    features=[col]
                ))
        
        return reports
    
    def _detect_chi2(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Chi-square test"""
        reports = []
        threshold = self.config.thresholds.get('chi2', 0.05)
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'categorical':
                continue
            
            ref_counts = self.reference_stats[col]['value_counts']
            new_counts = data[col].value_counts()
            
            # Align categories
            all_categories = sorted(
                set(ref_counts.index) | set(new_counts.index)
            )
            
            ref_freq = np.array([
                ref_counts.get(cat, 0) for cat in all_categories
            ])
            new_freq = np.array([
                new_counts.get(cat, 0) for cat in all_categories
            ])
            
            # Chi-square test requires at least 5 samples per category
            if min(ref_freq) >= 5 and min(new_freq) >= 5:
                statistic, p_value = stats.chi2_contingency(
                    [ref_freq, new_freq]
                )[:2]
                
                reports.append(DriftReport(
                    method='chi2',
                    score=p_value,  # Using p-value as score
                    threshold=threshold,
                    features=[col]
                ))
        
        return reports
