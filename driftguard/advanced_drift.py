import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest

@dataclass
class DriftMetrics:
    """Container for drift detection metrics"""
    p_value: float
    drift_score: float
    importance_score: float
    change_direction: str
    is_significant: bool

class AdvancedDriftDetector:
    """Advanced drift detection with multiple statistical methods and feature importance"""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        significance_level: float = 0.05,
        window_size: int = 100,
        feature_importance_threshold: float = 0.1
    ):
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.window_size = window_size
        self.feature_importance_threshold = feature_importance_threshold
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(reference_data)
        
    def detect_multivariate_drift(
        self,
        new_data: pd.DataFrame,
        method: str = "hotelling"
    ) -> Dict[str, DriftMetrics]:
        """
        Detect multivariate drift using various statistical methods.
        
        Args:
            new_data: New data to compare against reference
            method: Statistical method ('hotelling', 'mmd', 'isolation_forest')
            
        Returns:
            Dictionary containing drift metrics for each feature
        """
        drift_results = {}
        
        # Calculate feature importance scores
        importance_scores = self._calculate_feature_importance(new_data)
        
        for column in new_data.columns:
            if method == "hotelling":
                drift_metrics = self._hotelling_test(
                    self.reference_data[column],
                    new_data[column]
                )
            elif method == "mmd":
                drift_metrics = self._maximum_mean_discrepancy(
                    self.reference_data[column],
                    new_data[column]
                )
            else:  # isolation_forest
                drift_metrics = self._isolation_forest_drift(
                    new_data[column]
                )
                
            drift_metrics.importance_score = importance_scores.get(column, 0.0)
            drift_results[column] = drift_metrics
            
        return drift_results
    
    def detect_temporal_drift(
        self,
        time_series_data: pd.DataFrame,
        timestamp_column: str
    ) -> Dict[str, List[DriftMetrics]]:
        """
        Detect drift in time series data with seasonal adjustment.
        
        Args:
            time_series_data: DataFrame with timestamp column
            timestamp_column: Name of the timestamp column
            
        Returns:
            Dictionary containing drift metrics over time for each feature
        """
        drift_over_time = {}
        
        # Sort by timestamp
        time_series_data = time_series_data.sort_values(timestamp_column)
        
        for column in time_series_data.columns:
            if column == timestamp_column:
                continue
                
            # Apply seasonal decomposition
            try:
                series = pd.Series(
                    time_series_data[column].values,
                    index=time_series_data[timestamp_column]
                )
                drift_metrics = self._detect_seasonal_drift(series)
                drift_over_time[column] = drift_metrics
            except Exception as e:
                warnings.warn(f"Could not process temporal drift for {column}: {str(e)}")
                
        return drift_over_time
    
    def _hotelling_test(
        self,
        reference: pd.Series,
        new_data: pd.Series
    ) -> DriftMetrics:
        """Hotelling's T-squared test for drift detection"""
        ref_mean = reference.mean()
        new_mean = new_data.mean()
        
        # Calculate pooled covariance
        n1, n2 = len(reference), len(new_data)
        s1, s2 = reference.var(), new_data.var()
        pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        
        # Calculate T-squared statistic
        t_squared = (ref_mean - new_mean) ** 2 / (pooled_var * (1/n1 + 1/n2))
        p_value = 1 - stats.f.cdf(t_squared, 1, n1 + n2 - 2)
        
        return DriftMetrics(
            p_value=p_value,
            drift_score=1 - p_value,
            importance_score=0.0,  # Will be set later
            change_direction="increase" if new_mean > ref_mean else "decrease",
            is_significant=p_value < self.significance_level
        )
    
    def _maximum_mean_discrepancy(
        self,
        reference: pd.Series,
        new_data: pd.Series
    ) -> DriftMetrics:
        """Maximum Mean Discrepancy (MMD) test for drift detection"""
        X = reference.values.reshape(-1, 1)
        Y = new_data.values.reshape(-1, 1)
        
        # Compute kernel matrices
        XX = self._rbf_kernel(X, X)
        YY = self._rbf_kernel(Y, Y)
        XY = self._rbf_kernel(X, Y)
        
        # Calculate MMD
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        
        # Approximate p-value using permutation test
        n_permutations = 100
        pooled = np.concatenate([X, Y])
        mmd_permutations = []
        
        for _ in range(n_permutations):
            np.random.shuffle(pooled)
            perm_X = pooled[:len(X)]
            perm_Y = pooled[len(X):]
            XX_perm = self._rbf_kernel(perm_X, perm_X)
            YY_perm = self._rbf_kernel(perm_Y, perm_Y)
            XY_perm = self._rbf_kernel(perm_X, perm_Y)
            mmd_permutations.append(
                XX_perm.mean() + YY_perm.mean() - 2 * XY_perm.mean()
            )
            
        p_value = np.mean(np.array(mmd_permutations) >= mmd)
        
        return DriftMetrics(
            p_value=p_value,
            drift_score=mmd,
            importance_score=0.0,  # Will be set later
            change_direction="unknown",
            is_significant=p_value < self.significance_level
        )
    
    def _isolation_forest_drift(
        self,
        new_data: pd.Series
    ) -> DriftMetrics:
        """Use Isolation Forest for drift detection"""
        anomaly_scores = -self.isolation_forest.score_samples(
            new_data.values.reshape(-1, 1)
        )
        drift_score = np.mean(anomaly_scores)
        p_value = np.mean(anomaly_scores <= self.feature_importance_threshold)
        
        return DriftMetrics(
            p_value=p_value,
            drift_score=drift_score,
            importance_score=0.0,  # Will be set later
            change_direction="unknown",
            is_significant=drift_score > self.feature_importance_threshold
        )
    
    def _detect_seasonal_drift(
        self,
        series: pd.Series
    ) -> List[DriftMetrics]:
        """Detect drift in seasonal time series data"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            series,
            period=self._estimate_seasonality(series)
        )
        
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Analyze components
        drift_metrics = []
        window = self.window_size
        
        for i in range(window, len(series), window):
            window_data = series.iloc[i-window:i]
            next_window = series.iloc[i:i+window]
            
            if len(next_window) < window:
                break
                
            metrics = self._hotelling_test(window_data, next_window)
            drift_metrics.append(metrics)
            
        return drift_metrics
    
    def _calculate_feature_importance(
        self,
        new_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate feature importance scores based on drift impact"""
        importance_scores = {}
        
        for column in new_data.columns:
            ref_dist = self.reference_data[column]
            new_dist = new_data[column]
            
            # Calculate distribution difference
            if ref_dist.dtype.kind in 'fc':  # float or complex
                score = abs(ref_dist.mean() - new_dist.mean()) / ref_dist.std()
            else:  # categorical
                ref_freq = ref_dist.value_counts(normalize=True)
                new_freq = new_dist.value_counts(normalize=True)
                score = np.sum(abs(ref_freq - new_freq.reindex_like(ref_freq).fillna(0)))
                
            importance_scores[column] = score
            
        # Normalize scores
        max_score = max(importance_scores.values())
        if max_score > 0:
            importance_scores = {
                k: v/max_score for k, v in importance_scores.items()
            }
            
        return importance_scores
    
    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """RBF kernel for MMD calculation"""
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        gamma = 1.0 / X.shape[1]
        K = np.exp(-gamma * (X_norm + Y_norm - 2 * np.dot(X, Y.T)))
        return K
    
    @staticmethod
    def _estimate_seasonality(series: pd.Series) -> int:
        """Estimate seasonality period using autocorrelation"""
        n = len(series)
        if n < 4:  # Too short for seasonal detection
            return 1
            
        acf = np.correlate(series - series.mean(), series - series.mean(), mode='full')
        acf = acf[n-1:] / acf[n-1]
        
        # Find first peak after lag 1
        peaks = np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
        if len(peaks) > 0:
            return peaks[0]
        return 1  # No clear seasonality detected
