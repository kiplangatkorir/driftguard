"""
Drift detection module for DriftGuard.
"""
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from collections import defaultdict
import logging
import shap
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig

logger = logging.getLogger(__name__)

# Thread pool sizing constants
DEFAULT_CPU_COUNT = 4  # Fallback when os.cpu_count() returns None
WORKER_SCALE_FACTOR = 2  # Multiplier for initial worker count
MAX_WORKER_SCALE_FACTOR = 4  # Upper bound multiplier for worker count

class DriftDetector(IDriftDetector):
    """Detects data drift using multiple statistical methods"""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """Initialize drift detector"""
        self.config = config or DriftConfig()
        self.reference_data = None
        self.feature_types = {}
        self.reference_stats = {}
        self._initialized = False
        self._explainer = None
        self._baseline_shap = None
        self.model = None
        
        # Initialize caches
        self._stats_cache = {}
        self._shap_cache = {}
        
    def _cache_key(self, data: pd.DataFrame, col: str, method: str) -> str:
        """Generate cache key for statistical tests"""
        return f"{method}_{col}_{hash(frozenset(data[col].values.tobytes()))}"
        
    def _cached_stat_test(self, data: pd.DataFrame, col: str, method: str, test_func):
        """Run statistical test with caching"""
        key = self._cache_key(data, col, method)
        
        if key in self._stats_cache:
            return self._stats_cache[key]
            
        result = test_func(data[col].values, self.reference_data[col].values)
        self._stats_cache[key] = result
        return result
    
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
        
        # Group features by type for batch processing
        self._feature_groups = self._group_features_by_type()
        
        # Initialize SHAP explainer if model is available
        if self.model is not None:
            self._explainer = shap.Explainer(self.model.predict_proba, reference_data)
            self._baseline_shap = self._calculate_shap_values(self.reference_data)
        
        self._initialized = True
    
    def attach_model(self, model):
        """Attach a model for feature importance analysis.
        
        Warning: SHAP explainers may not be thread-safe. When using multithreaded
        drift detection (via detect() method), SHAP-based importance calculations
        may experience race conditions. Consider disabling multithreading 
        (set auto_scale_workers=False, max_workers=1) if SHAP stability issues occur.
        """
        self.model = model
        if self._initialized and self.reference_data is not None:
            self._explainer = shap.Explainer(model.predict_proba, self.reference_data)
            self._baseline_shap = self._calculate_shap_values(self.reference_data)
    
    def _calculate_shap_values(self, data: pd.DataFrame):
        """Optimized SHAP calculation with caching"""
        cache_key = hash(frozenset(data.values.tobytes()))
        
        if hasattr(self, '_shap_cache') and cache_key in self._shap_cache:
            return self._shap_cache[cache_key]
            
        if not hasattr(self, '_shap_cache'):
            self._shap_cache = {}
            
        # Use approximate SHAP for faster computation
        shap_values = self._explainer(data, check_additivity=False)
        self._shap_cache[cache_key] = shap_values
        
        return shap_values
    
    def _group_features_by_type(self) -> Dict[str, List[str]]:
        """Group features by their type for batch processing"""
        grouped = {'categorical': [], 'continuous': []}
        for col, ftype in self.feature_types.items():
            grouped[ftype].append(col)
        return grouped
    
    def _detect_feature(self, data: pd.DataFrame, col: str, method: str) -> DriftReport:
        """Detect drift for a single feature"""
        if method == 'ks':
            if self.feature_types[col] != 'continuous':
                return None
            
            ref_values = self.reference_data[col].dropna()
            new_values = data[col].dropna()
            
            if len(new_values) < 2:
                return None
            
            statistic, _ = self._cached_stat_test(data, col, method, stats.ks_2samp)
            
            # Calculate importance change if SHAP values available
            importance_change = None
            if self._explainer is not None and col in self.reference_data.columns:
                col_idx = data.columns.get_loc(col)
                shap_values = self._calculate_shap_values(data)
                importance_change = (
                    np.abs(shap_values.values[:,col_idx]).mean() - 
                    np.abs(self._baseline_shap.values[:,col_idx]).mean()
                )
                
            return DriftReport(
                method='ks',
                score=statistic,
                threshold=self.config.thresholds['ks'],
                features=[col],
                importance_change=importance_change
            )
        elif method == 'jsd':
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
            
            # Calculate importance change if SHAP values available
            importance_change = None
            if self._explainer is not None and col in self.reference_data.columns:
                col_idx = data.columns.get_loc(col)
                shap_values = self._calculate_shap_values(data)
                importance_change = (
                    np.abs(shap_values.values[:,col_idx]).mean() - 
                    np.abs(self._baseline_shap.values[:,col_idx]).mean()
                )
                
            return DriftReport(
                method='jsd',
                score=jsd,
                threshold=self.config.thresholds['jsd'],
                features=[col],
                importance_change=importance_change
            )
        elif method == 'psi':
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
            
            # Calculate importance change if SHAP values available
            importance_change = None
            if self._explainer is not None and col in self.reference_data.columns:
                col_idx = data.columns.get_loc(col)
                shap_values = self._calculate_shap_values(data)
                importance_change = (
                    np.abs(shap_values.values[:,col_idx]).mean() - 
                    np.abs(self._baseline_shap.values[:,col_idx]).mean()
                )
                
            return DriftReport(
                method='psi',
                score=psi,
                threshold=self.config.thresholds['psi'],
                features=[col],
                importance_change=importance_change
            )
        elif method == 'wasserstein':
            if self.feature_types[col] != 'continuous':
                return None
            
            ref_values = self.reference_data[col].dropna()
            new_values = data[col].dropna()
            
            if len(new_values) < 2:
                return None
            
            # Normalize values to [0,1] range
            ref_min = ref_values.min()
            ref_max = ref_values.max()
            if ref_max > ref_min:
                ref_norm = (ref_values - ref_min) / (ref_max - ref_min)
                new_norm = (new_values - ref_min) / (ref_max - ref_min)
                
                distance = wasserstein_distance(ref_norm, new_norm)
                
                # Calculate importance change if SHAP values available
                importance_change = None
                if self._explainer is not None and col in self.reference_data.columns:
                    col_idx = data.columns.get_loc(col)
                    shap_values = self._calculate_shap_values(data)
                    importance_change = (
                        np.abs(shap_values.values[:,col_idx]).mean() - 
                        np.abs(self._baseline_shap.values[:,col_idx]).mean()
                    )
                    
                return DriftReport(
                    method='wasserstein',
                    score=distance,
                    threshold=self.config.thresholds.get('wasserstein', 0.1),
                    features=[col],
                    importance_change=importance_change
                )
        elif method == 'chi2':
            if self.feature_types[col] != 'categorical':
                return None
            
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
                statistic, p_value = self._cached_stat_test(data, col, method, stats.chi2_contingency)
                
                # Calculate importance change if SHAP values available
                importance_change = None
                if self._explainer is not None and col in self.reference_data.columns:
                    col_idx = data.columns.get_loc(col)
                    shap_values = self._calculate_shap_values(data)
                    importance_change = (
                        np.abs(shap_values.values[:,col_idx]).mean() - 
                        np.abs(self._baseline_shap.values[:,col_idx]).mean()
                    )
                    
                return DriftReport(
                    method='chi2',
                    score=p_value,  # Using p-value as score
                    threshold=self.config.thresholds.get('chi2', 0.05),
                    features=[col],
                    importance_change=importance_change
                )
    
    def _group_features_by_type(self) -> Dict[str, List[str]]:
        """Group features by their data type"""
        grouped = {'continuous': [], 'categorical': []}
        for col, ftype in self.feature_types.items():
            grouped[ftype].append(col)
        return grouped
    
    def _process_features_batch(self, batch: pd.DataFrame, features: List[str], method: str) -> List[DriftReport]:
        """Process multiple features of the same type together
        
        This method processes features sequentially but groups them by type for better
        organization and potential future optimizations. The feature_batch_size config
        parameter is reserved for future chunking optimizations.
        """
        reports = []
        for col in features:
            report = self._detect_feature(batch, col, method)
            if report is not None:
                reports.append(report)
        return reports
    
    def _process_batch(self, batch: pd.DataFrame) -> List[DriftReport]:
        """Process a batch of data with grouped feature processing"""
        reports = []
        
        # Check if batch processing is enabled
        if not self.config.batch_feature_processing:
            # Fall back to original implementation
            total = len(self.config.methods) * len(self.reference_data.columns)
            
            with tqdm(total=total, desc="Processing features") as pbar:
                for method in self.config.methods:
                    for col in self.reference_data.columns:
                        report = self._detect_feature(batch, col, method)
                        if report is not None:
                            reports.append(report)
                        pbar.update(1)
            
            return reports
        
        # Use grouped feature processing
        feature_groups = self._group_features_by_type()
        
        total = len(self.config.methods) * len(self.reference_data.columns)
        
        with tqdm(total=total, desc="Processing features") as pbar:
            for method in self.config.methods:
                # Process continuous features in batch
                if feature_groups['continuous']:
                    continuous_reports = self._process_features_batch(
                        batch, feature_groups['continuous'], method
                    )
                    reports.extend(continuous_reports)
                    pbar.update(len(feature_groups['continuous']))
                
                # Process categorical features in batch
                if feature_groups['categorical']:
                    categorical_reports = self._process_features_batch(
                        batch, feature_groups['categorical'], method
                    )
                    reports.extend(categorical_reports)
                    pbar.update(len(feature_groups['categorical']))
        
        return reports
    
    def _process_continuous_features(
        self, 
        batch: pd.DataFrame, 
        features: List[str], 
        method: str
    ) -> List[DriftReport]:
        """Process continuous features in batch"""
        reports = []
        for col in features:
            report = self._detect_feature(batch, col, method)
            if report is not None:
                reports.append(report)
        return reports
    
    def _process_categorical_features(
        self, 
        batch: pd.DataFrame, 
        features: List[str], 
        method: str
    ) -> List[DriftReport]:
        """Process categorical features in batch"""
        reports = []
        for col in features:
            report = self._detect_feature(batch, col, method)
            if report is not None:
                reports.append(report)
        return reports
    
    def detect(self, data: pd.DataFrame, batch_size: int = 1000) -> List[DriftReport]:
        """Process data in memory-efficient batches with adaptive threading"""
        if not self._initialized:
            raise RuntimeError("Detector not initialized")
            
        results = []
        batches = [data.iloc[i:i+batch_size] 
                  for i in range(0, len(data), batch_size)]
        
        # Handle empty input
        if not batches:
            return []
        
        # Adaptive thread pool sizing
        cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT
        if self.config.max_workers is not None:
            max_workers = self.config.max_workers
        elif self.config.auto_scale_workers:
            # Scale based on workload and CPU
            max_workers = min(
                cpu_count * WORKER_SCALE_FACTOR,
                len(batches),
                cpu_count * MAX_WORKER_SCALE_FACTOR
            )
        else:
            max_workers = cpu_count
        
        logger.info("Using %d workers for drift detection", max_workers)
                  
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_batch, batch) 
                      for batch in batches]
            for future in as_completed(futures):
                results.extend(future.result())
                
        return results
