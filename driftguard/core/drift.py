"""
Enhanced drift detection module for DriftGuard v0.1.5
Includes caching, vectorization, and advanced statistical methods.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from scipy.special import rel_entr
from collections import defaultdict
import logging
from functools import lru_cache
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig

logger = logging.getLogger(__name__)

@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    mean: float
    std: float
    hist: np.ndarray
    quantiles: np.ndarray
    n_samples: int
    value_counts: Optional[pd.Series] = None

class DriftDetector(IDriftDetector):
    """Enhanced drift detector using multiple statistical methods with caching and vectorization"""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """Initialize drift detector with enhanced configuration"""
        self.config = config or DriftConfig()
        self.reference_data = None
        self.feature_types = {}
        self.reference_stats: Dict[str, FeatureStats] = {}
        self._initialized = False
        self._cache = {}
        self.max_workers = self.config.max_workers or 4
    
    @lru_cache(maxsize=128)
    def _compute_feature_stats(self, feature_name: str, values: np.ndarray) -> FeatureStats:
        """Compute and cache feature statistics"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            stats = FeatureStats(
                mean=np.mean(values),
                std=np.std(values),
                hist=np.histogram(values, bins=20)[0],
                quantiles=np.percentile(values, [25, 50, 75]),
                n_samples=len(values)
            )
            if self.feature_types[feature_name] == 'categorical':
                stats.value_counts = pd.Series(values).value_counts(normalize=True)
            return stats

    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize detector with reference data using parallel processing"""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
        
        self.reference_data = reference_data.copy()
        
        # Determine feature types with improved type inference
        self.feature_types = {
            col: 'categorical' if (
                pd.api.types.is_categorical_dtype(reference_data[col]) or
                pd.api.types.is_object_dtype(reference_data[col]) or
                reference_data[col].nunique() / len(reference_data) < 0.05  # Auto-detect categorical
            ) else 'continuous'
            for col in reference_data.columns
        }
        
        # Compute reference statistics in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                col: executor.submit(
                    self._compute_feature_stats,
                    col,
                    reference_data[col].dropna().values
                )
                for col in reference_data.columns
            }
            self.reference_stats = {
                col: future.result()
                for col, future in futures.items()
            }
        
        self._initialized = True

    async def detect_async(self, data: pd.DataFrame) -> List[DriftReport]:
        """Asynchronous drift detection"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.detect, data
        )

    def detect(self, data: pd.DataFrame) -> List[DriftReport]:
        """Enhanced drift detection with parallel processing"""
        if not self._initialized:
            raise ValueError("Detector not initialized")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        reports = []
        
        # Detect drift for each feature using configured methods in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for method in self.config.methods:
                detect_func = getattr(self, f'_detect_{method}')
                futures.append(executor.submit(detect_func, data))
            
            for future in futures:
                reports.extend(future.result())
        
        return reports

    def _detect_ks(self, data: pd.DataFrame) -> List[DriftReport]:
        """Vectorized Kolmogorov-Smirnov test"""
        reports = []
        threshold = self.config.thresholds['ks']
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'continuous':
                continue
            
            ref_values = self.reference_data[col].dropna().values
            new_values = data[col].dropna().values
            
            if len(new_values) < 2:
                continue
            
            # Vectorized KS test
            statistic, pvalue = stats.ks_2samp(ref_values, new_values)
            
            if pvalue < threshold:
                reports.append(DriftReport(
                    feature=col,
                    method='ks',
                    statistic=float(statistic),
                    pvalue=float(pvalue),
                    threshold=threshold,
                    drift_detected=True
                ))
        
        return reports

    def _detect_anderson(self, data: pd.DataFrame) -> List[DriftReport]:
        """Anderson-Darling test for drift detection"""
        reports = []
        threshold = self.config.thresholds.get('anderson', 0.05)
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'continuous':
                continue
            
            ref_values = self.reference_data[col].dropna().values
            new_values = data[col].dropna().values
            
            if len(new_values) < 2:
                continue
            
            statistic, critical_values, significance_level = stats.anderson_ksamp([ref_values, new_values])
            
            # Compare with critical value at 5% significance level
            critical_value = critical_values[2]  # 5% significance level
            drift_detected = statistic > critical_value
            
            if drift_detected:
                reports.append(DriftReport(
                    feature=col,
                    method='anderson',
                    statistic=float(statistic),
                    pvalue=significance_level,
                    threshold=threshold,
                    drift_detected=True,
                    metadata={'critical_value': float(critical_value)}
                ))
        
        return reports

    def _detect_wasserstein(self, data: pd.DataFrame) -> List[DriftReport]:
        """Vectorized Wasserstein distance calculation"""
        reports = []
        threshold = self.config.thresholds['wasserstein']
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'continuous':
                continue
            
            ref_values = self.reference_data[col].dropna().values
            new_values = data[col].dropna().values
            
            if len(new_values) < 2:
                continue
            
            # Normalize values for fair comparison
            ref_normalized = (ref_values - np.mean(ref_values)) / np.std(ref_values)
            new_normalized = (new_values - np.mean(new_values)) / np.std(new_values)
            
            distance = wasserstein_distance(ref_normalized, new_normalized)
            
            if distance > threshold:
                reports.append(DriftReport(
                    feature=col,
                    method='wasserstein',
                    statistic=float(distance),
                    threshold=threshold,
                    drift_detected=True
                ))
        
        return reports

    def _detect_concept_drift(self, data: pd.DataFrame, target: str, predictions: np.ndarray) -> List[DriftReport]:
        """Detect concept drift using performance metrics"""
        if target not in data.columns:
            return []
        
        reports = []
        threshold = self.config.thresholds.get('concept_drift', 0.1)
        
        # Calculate performance metrics
        y_true = data[target].values
        
        if self.feature_types[target] == 'continuous':
            # Regression metrics
            mse = np.mean((y_true - predictions) ** 2)
            ref_mse = self.reference_stats.get('mse', None)
            
            if ref_mse and abs(mse - ref_mse) > threshold:
                reports.append(DriftReport(
                    feature='model_performance',
                    method='concept_drift',
                    statistic=float(abs(mse - ref_mse)),
                    threshold=threshold,
                    drift_detected=True,
                    metadata={'metric': 'mse', 'current': mse, 'reference': ref_mse}
                ))
        else:
            # Classification metrics
            accuracy = np.mean(y_true == predictions)
            ref_accuracy = self.reference_stats.get('accuracy', None)
            
            if ref_accuracy and abs(accuracy - ref_accuracy) > threshold:
                reports.append(DriftReport(
                    feature='model_performance',
                    method='concept_drift',
                    statistic=float(abs(accuracy - ref_accuracy)),
                    threshold=threshold,
                    drift_detected=True,
                    metadata={'metric': 'accuracy', 'current': accuracy, 'reference': ref_accuracy}
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
                ref_hist = self.reference_stats[col].hist
                new_hist = np.histogram(
                    data[col].dropna(),
                    bins=20
                )[0]
            else:
                ref_counts = self.reference_stats[col].value_counts
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
                feature=col,
                method='jsd',
                statistic=float(jsd),
                threshold=threshold,
                drift_detected=jsd > threshold
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
                ref_hist = self.reference_stats[col].hist
                new_hist = np.histogram(
                    data[col].dropna(),
                    bins=20
                )[0]
            else:
                ref_counts = self.reference_stats[col].value_counts
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
                feature=col,
                method='psi',
                statistic=float(psi),
                threshold=threshold,
                drift_detected=psi > threshold
            ))
        
        return reports

    def _detect_chi2(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift using Chi-square test"""
        reports = []
        threshold = self.config.thresholds.get('chi2', 0.05)
        
        for col in self.reference_data.columns:
            if self.feature_types[col] != 'categorical':
                continue
            
            ref_counts = self.reference_stats[col].value_counts
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
                    feature=col,
                    method='chi2',
                    statistic=float(statistic),
                    pvalue=float(p_value),
                    threshold=threshold,
                    drift_detected=p_value < threshold
                ))
        
        return reports
