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
from sklearn.preprocessing import StandardScaler
from river import drift as river_drift
import joblib
from functools import lru_cache

from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig

logger = logging.getLogger(__name__)

class DriftDetector(IDriftDetector):
    """Detects data drift using multiple statistical methods including multivariate analysis"""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """Initialize drift detector with enhanced capabilities"""
        self.config = config or DriftConfig()
        self.reference_data = None
        self.feature_types = {}
        self.reference_stats = {}
        self._initialized = False
        self.scaler = StandardScaler()
        self.adwin_detectors = {}
        self._cache = {}
        
    @lru_cache(maxsize=128)
    def _compute_feature_stats(self, feature_name: str, data: np.ndarray) -> Dict:
        """Cached computation of feature statistics"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'quantiles': np.percentile(data, [25, 50, 75])
        }
        
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize detector with reference data and prepare multivariate analysis"""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
            
        self.reference_data = reference_data.copy()
        self.feature_types = self._infer_feature_types(reference_data)
        
        # Prepare multivariate analysis
        numerical_cols = [col for col, type_ in self.feature_types.items() if type_ == 'numerical']
        if numerical_cols:
            self.scaler.fit(reference_data[numerical_cols])
            
        # Initialize ADWIN detectors for each feature
        for feature in reference_data.columns:
            self.adwin_detectors[feature] = river_drift.ADWIN()
            
        self._initialized = True
        
    def detect_multivariate_drift(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift using multivariate analysis"""
        numerical_cols = [col for col, type_ in self.feature_types.items() if type_ == 'numerical']
        if not numerical_cols:
            return {}
            
        ref_scaled = self.scaler.transform(self.reference_data[numerical_cols])
        new_scaled = self.scaler.transform(new_data[numerical_cols])
        
        # Compute Mahalanobis distance
        ref_cov = np.cov(ref_scaled.T)
        ref_mean = np.mean(ref_scaled, axis=0)
        
        distances = []
        for sample in new_scaled:
            diff = sample - ref_mean
            dist = np.sqrt(diff.dot(np.linalg.inv(ref_cov)).dot(diff))
            distances.append(dist)
            
        return {
            'mahalanobis_mean': np.mean(distances),
            'mahalanobis_threshold': np.percentile(distances, 95)
        }
        
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
            elif method == 'adwin':
                reports.extend(self._detect_adwin(data))
            elif method == 'multivariate':
                reports.extend(self._detect_multivariate(data))
        
        return reports
    
    def _infer_feature_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Infer feature types from data"""
        feature_types = {}
        for col in data.columns:
            if pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                feature_types[col] = 'categorical'
            elif pd.api.types.is_numeric_dtype(data[col]):
                feature_types[col] = 'numerical'
            else:
                raise ValueError(f"Unsupported data type for feature {col}")
        return feature_types
    
    def _detect_ks(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Kolmogorov-Smirnov test"""
        reports = []
        threshold = self.config.thresholds['ks']
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'numerical':
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
            
            if self.feature_types[col] == 'numerical':
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
            
            if self.feature_types[col] == 'numerical':
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
            if self.feature_types[col] != 'numerical':
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
    
    def _detect_adwin(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using ADWIN"""
        reports = []
        threshold = self.config.thresholds.get('adwin', 0.05)
        
        for col in self.reference_data.columns:
            detector = self.adwin_detectors[col]
            values = data[col].dropna()
            
            for value in values:
                detector.update(value)
                
                if detector.drift_detected:
                    reports.append(DriftReport(
                        method='adwin',
                        score=detector.estimation,
                        threshold=threshold,
                        features=[col]
                    ))
        
        return reports
    
    def _detect_multivariate(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using multivariate analysis"""
        reports = []
        threshold = self.config.thresholds.get('multivariate', 0.05)
        
        result = self.detect_multivariate_drift(data)
        
        if result:
            reports.append(DriftReport(
                method='multivariate',
                score=result['mahalanobis_mean'],
                threshold=threshold,
                features=list(data.columns)
            ))
        
        return reports
