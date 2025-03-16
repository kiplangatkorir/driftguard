"""
Advanced drift detection system for DriftGuard.
Provides multiple drift detection algorithms with extensible architecture.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mutual_info_score
import warnings
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .interfaces import IDriftDetector, DriftReport
from .config import DriftConfig, DriftMethod

logger = logging.getLogger(__name__)

class BaseDriftDetector(IDriftDetector, ABC):
    """Base class for drift detection implementations"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.reference_data = None
        self.feature_importance = {}
        self._initialize_detector()
    
    @abstractmethod
    def _initialize_detector(self) -> None:
        """Initialize specific detector implementation"""
        pass
    
    @abstractmethod
    def _compute_drift_score(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """Compute drift score and p-value"""
        pass
    
    def initialize(self, reference_data: pd.DataFrame, config: Optional[DriftConfig] = None) -> None:
        """Initialize detector with reference data"""
        if config:
            self.config = config
            
        self.reference_data = reference_data
        self._compute_feature_importance()
    
    def _compute_feature_importance(self) -> None:
        """Compute feature importance scores"""
        if self.reference_data is None:
            return
            
        importance = {}
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['int64', 'float64']:
                # Use standard deviation for numerical features
                importance[col] = float(self.reference_data[col].std())
            else:
                # Use entropy for categorical features
                value_counts = self.reference_data[col].value_counts(normalize=True)
                importance[col] = float(-np.sum(value_counts * np.log2(value_counts)))
                
        # Normalize importance scores
        max_importance = max(importance.values())
        self.feature_importance = {
            k: v/max_importance for k, v in importance.items()
        }
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        if self.reference_data is None:
            raise ValueError("Detector not initialized with reference data")
            
        reports = []
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                continue
                
            ref_data = self.reference_data[column].values
            new_data_col = new_data[column].values
            
            try:
                drift_score, p_value = self._compute_drift_score(ref_data, new_data_col)
                
                reports.append(DriftReport(
                    feature_name=column,
                    drift_score=drift_score,
                    p_value=p_value,
                    timestamp=datetime.now(),
                    sample_size=len(new_data_col),
                    drift_type=self.__class__.__name__,
                    additional_metrics={
                        "importance_score": self.feature_importance.get(column, 0.0)
                    }
                ))
            except Exception as e:
                logger.error(f"Failed to compute drift for {column}: {str(e)}")
                
        return reports
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        self.reference_data = new_reference
        self._compute_feature_importance()
        self._initialize_detector()

class KSTestDriftDetector(BaseDriftDetector):
    """Drift detector using Kolmogorov-Smirnov test"""
    
    def _initialize_detector(self) -> None:
        pass  # No initialization needed for KS test
    
    def _compute_drift_score(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        if reference.dtype.kind in 'fc' and current.dtype.kind in 'fc':
            statistic, p_value = stats.ks_2samp(reference, current)
            return float(statistic), float(p_value)
        else:
            # For categorical data, compare distributions
            ref_dist = pd.Series(reference).value_counts(normalize=True)
            cur_dist = pd.Series(current).value_counts(normalize=True)
            
            # Align distributions
            ref_dist, cur_dist = ref_dist.align(cur_dist, fill_value=0)
            
            # Compute Jensen-Shannon divergence
            m = 0.5 * (ref_dist + cur_dist)
            js_div = 0.5 * (
                stats.entropy(ref_dist, m) +
                stats.entropy(cur_dist, m)
            )
            
            # Convert to p-value using chi-square approximation
            p_value = 1 - stats.chi2.cdf(js_div, df=1)
            return float(js_div), float(p_value)

class IsolationForestDriftDetector(BaseDriftDetector):
    """Drift detector using Isolation Forest"""
    
    def _initialize_detector(self) -> None:
        """Initialize Isolation Forest"""
        if self.reference_data is not None:
            self.detector = IsolationForest(
                contamination=self.config.threshold,
                random_state=42
            )
            # Fit on reference data
            numerical_cols = self.reference_data.select_dtypes(
                include=['int64', 'float64']
            ).columns
            if not numerical_cols.empty:
                self.detector.fit(self.reference_data[numerical_cols])
    
    def _compute_drift_score(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        if reference.dtype.kind not in 'fc' or current.dtype.kind not in 'fc':
            return 0.0, 1.0  # Skip non-numerical data
            
        # Compute anomaly scores
        current_scores = -self.detector.score_samples(
            current.reshape(-1, 1)
        )
        reference_scores = -self.detector.score_samples(
            reference.reshape(-1, 1)
        )
        
        # Compare score distributions
        statistic, p_value = stats.ks_2samp(reference_scores, current_scores)
        return float(statistic), float(p_value)

class MMDDriftDetector(BaseDriftDetector):
    """Drift detector using Maximum Mean Discrepancy"""
    
    def _initialize_detector(self) -> None:
        pass  # No initialization needed for MMD
    
    def _compute_kernel(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sigma: Optional[float] = None
    ) -> np.ndarray:
        """Compute RBF kernel matrix"""
        if sigma is None:
            # Median heuristic for sigma
            X_flat = X.ravel()
            Y_flat = Y.ravel()
            distances = np.abs(X_flat.reshape(-1, 1) - Y_flat.reshape(1, -1))
            sigma = np.median(distances)
            sigma = max(sigma, 1e-10)  # Prevent sigma = 0
            
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
        
        dnorm2 = (
            XX.diagonal().reshape(-1, 1) +
            YY.diagonal().reshape(1, -1) -
            2 * XY
        )
        
        return np.exp(-dnorm2 / (2 * sigma * sigma))
    
    def _compute_drift_score(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        if reference.dtype.kind not in 'fc' or current.dtype.kind not in 'fc':
            return 0.0, 1.0  # Skip non-numerical data
            
        # Reshape and normalize data
        X = reference.reshape(-1, 1)
        Y = current.reshape(-1, 1)
        
        # Standardize
        X = (X - X.mean()) / (X.std() + 1e-10)
        Y = (Y - Y.mean()) / (Y.std() + 1e-10)
        
        # Compute kernel matrices
        K_XX = self._compute_kernel(X, X)
        K_YY = self._compute_kernel(Y, Y)
        K_XY = self._compute_kernel(X, Y)
        
        # Compute MMD
        mmd = (
            K_XX.mean() +
            K_YY.mean() -
            2 * K_XY.mean()
        )
        
        # Approximate p-value using permutation test
        n_permutations = 100
        pooled = np.concatenate([X, Y])
        permutation_mmd = []
        
        for _ in range(n_permutations):
            np.random.shuffle(pooled)
            perm_X = pooled[:len(X)]
            perm_Y = pooled[len(X):]
            
            K_XX_perm = self._compute_kernel(perm_X, perm_X)
            K_YY_perm = self._compute_kernel(perm_Y, perm_Y)
            K_XY_perm = self._compute_kernel(perm_X, perm_Y)
            
            perm_mmd = (
                K_XX_perm.mean() +
                K_YY_perm.mean() -
                2 * K_XY_perm.mean()
            )
            permutation_mmd.append(perm_mmd)
            
        p_value = np.mean(np.array(permutation_mmd) >= mmd)
        return float(mmd), float(p_value)

class AdwinDriftDetector(BaseDriftDetector):
    """Drift detector using Adaptive Windowing"""
    
    def _initialize_detector(self) -> None:
        self.window_size = self.config.window_size
        self.delta = self.config.threshold
        self.windows = {}  # Store windows for each feature
    
    def _compute_drift_score(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        # Combine data in temporal order
        combined = np.concatenate([reference[-self.window_size:], current])
        
        if len(combined) < 2 * self.window_size:
            return 0.0, 1.0
            
        # Split into two windows
        window1 = combined[:self.window_size]
        window2 = combined[-self.window_size:]
        
        # Compute means and variances
        mean1, var1 = window1.mean(), window1.var()
        mean2, var2 = window2.mean(), window2.var()
        
        # Compute Hoeffding bound
        n1 = len(window1)
        n2 = len(window2)
        bound = np.sqrt(
            (1.0 / (2 * n1) + 1.0 / (2 * n2)) *
            np.log(2.0 / self.delta)
        )
        
        # Compute drift score
        drift_score = abs(mean1 - mean2)
        p_value = 1.0 if drift_score <= bound else 0.0
        
        return float(drift_score), float(p_value)

def create_drift_detector(method: DriftMethod, config: DriftConfig) -> BaseDriftDetector:
    """Factory function to create drift detector"""
    detectors = {
        DriftMethod.KS_TEST: KSTestDriftDetector,
        DriftMethod.ISOLATION_FOREST: IsolationForestDriftDetector,
        DriftMethod.MMD: MMDDriftDetector,
        DriftMethod.ADWIN: AdwinDriftDetector
    }
    
    if method not in detectors:
        raise ValueError(f"Unsupported drift detection method: {method}")
        
    return detectors[method](config)
