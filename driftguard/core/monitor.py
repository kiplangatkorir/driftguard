"""
Enhanced model monitoring module for DriftGuard v0.1.5.
Includes advanced metrics, performance tracking, and ML explainability.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn import metrics
from datetime import datetime, timedelta
import logging
import shap
import lightgbm as lgb
from joblib import Parallel, delayed
import warnings
from dataclasses import dataclass
from collections import deque, defaultdict

from .interfaces import IModelMonitor
from .config import MonitorConfig
from .utils.data_utils import validate_inputs

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    value: float
    degraded: bool
    reference: float
    trend: str  # 'improving', 'stable', 'degrading'
    z_score: float
    timestamp: datetime

class ModelMonitor(IModelMonitor):
    """Enhanced model monitor with advanced metrics and ML explainability"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize enhanced model monitor"""
        self.config = config or MonitorConfig()
        self.reference_metrics = {}
        self.reference_predictions = None
        self.reference_labels = None
        self.feature_importance = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=self.config.history_size or 100))
        self.last_retrain_time = None
        self._initialized = False
        self._n_jobs = self.config.n_jobs or -1
    
    def initialize(
        self,
        reference_predictions: Union[pd.Series, np.ndarray],
        reference_labels: Union[pd.Series, np.ndarray],
        reference_features: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None
    ) -> None:
        """Initialize monitor with reference data and optional model for explainability"""
        validate_inputs(reference_predictions, reference_labels, "reference")
        
        self.reference_predictions = pd.Series(reference_predictions).copy()
        self.reference_labels = pd.Series(reference_labels).copy()
        
        # Compute reference metrics in parallel
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.reference_metrics = self._compute_metrics_parallel(
                self.reference_predictions,
                self.reference_labels
            )
        
        # Initialize feature importance if model and features provided
        if model is not None and reference_features is not None:
            self._initialize_feature_importance(model, reference_features)
        
        self.last_retrain_time = datetime.now()
        self._initialized = True
    
    def track(
        self,
        predictions: Union[pd.Series, np.ndarray],
        labels: Union[pd.Series, np.ndarray],
        features: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, PerformanceMetrics]:
        """Track model performance with enhanced metrics and feature importance"""
        if not self._initialized:
            raise ValueError("Monitor not initialized")
        
        validate_inputs(predictions, labels, "current")
        timestamp = timestamp or datetime.now()
        
        # Compute current metrics in parallel
        current_metrics = self._compute_metrics_parallel(
            pd.Series(predictions),
            pd.Series(labels)
        )
        
        # Update metric history
        for metric, value in current_metrics.items():
            self.metric_history[metric].append((timestamp, value))
        
        # Analyze performance and detect degradation
        metrics_with_status = self._analyze_performance(
            current_metrics,
            timestamp
        )
        
        # Check if retraining is needed
        if self._should_retrain(metrics_with_status):
            logger.warning("Model retraining recommended based on performance degradation")
        
        # Update feature importance if features provided
        if features is not None and hasattr(self, '_shap_explainer'):
            self._update_feature_importance(features)
        
        return metrics_with_status
    
    def _compute_metrics_parallel(
        self,
        predictions: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Compute metrics in parallel"""
        metrics_dict = {}
        
        def compute_metric(metric_name: str) -> Tuple[str, float]:
            if metric_name == 'accuracy':
                return metric_name, metrics.accuracy_score(labels, predictions)
            elif metric_name == 'precision':
                return metric_name, metrics.precision_score(labels, predictions, average='weighted')
            elif metric_name == 'recall':
                return metric_name, metrics.recall_score(labels, predictions, average='weighted')
            elif metric_name == 'f1':
                return metric_name, metrics.f1_score(labels, predictions, average='weighted')
            elif metric_name == 'roc_auc':
                return metric_name, metrics.roc_auc_score(labels, predictions)
            elif metric_name == 'pr_auc':
                precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
                return metric_name, metrics.auc(recall, precision)
            elif metric_name == 'mcc':
                return metric_name, metrics.matthews_corrcoef(labels, predictions)
            elif metric_name == 'brier':
                return metric_name, metrics.brier_score_loss(labels, predictions)
            elif metric_name == 'log_loss':
                return metric_name, metrics.log_loss(labels, predictions)
        
        # Compute metrics in parallel
        results = Parallel(n_jobs=self._n_jobs)(
            delayed(compute_metric)(metric)
            for metric in self.config.metrics
        )
        
        metrics_dict.update(dict(results))
        return metrics_dict
    
    def _analyze_performance(
        self,
        current_metrics: Dict[str, float],
        timestamp: datetime
    ) -> Dict[str, PerformanceMetrics]:
        """Analyze performance trends and detect degradation"""
        results = {}
        
        for metric, value in current_metrics.items():
            history = list(self.metric_history[metric])
            if len(history) >= 2:
                # Calculate trend using linear regression
                times = np.array([(t - history[0][0]).total_seconds() 
                                for t, _ in history])
                values = np.array([v for _, v in history])
                slope = np.polyfit(times, values, 1)[0]
                
                # Calculate z-score
                z_score = (value - np.mean(values)) / np.std(values)
                
                # Determine trend
                if abs(z_score) < 1:
                    trend = 'stable'
                else:
                    trend = 'improving' if slope > 0 else 'degrading'
            else:
                trend = 'stable'
                z_score = 0
            
            # Check for degradation
            degraded = self._is_metric_degraded(
                metric,
                value,
                self.reference_metrics[metric]
            )
            
            results[metric] = PerformanceMetrics(
                value=value,
                degraded=degraded,
                reference=self.reference_metrics[metric],
                trend=trend,
                z_score=z_score,
                timestamp=timestamp
            )
        
        return results
    
    def _initialize_feature_importance(
        self,
        model: Any,
        features: pd.DataFrame
    ) -> None:
        """Initialize SHAP explainer for feature importance tracking"""
        try:
            self._shap_explainer = shap.TreeExplainer(model)
            self._update_feature_importance(features)
        except Exception as e:
            logger.warning(f"Could not initialize feature importance: {str(e)}")
    
    def _update_feature_importance(self, features: pd.DataFrame) -> None:
        """Update feature importance using SHAP values"""
        try:
            shap_values = self._shap_explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_values = np.abs(shap_values).mean(axis=0)
            
            self.feature_importance = dict(zip(
                features.columns,
                shap_values / np.sum(shap_values)
            ))
        except Exception as e:
            logger.warning(f"Could not update feature importance: {str(e)}")
    
    def _is_metric_degraded(
        self,
        metric: str,
        current: float,
        reference: float
    ) -> bool:
        """Check if metric has degraded beyond threshold"""
        threshold = self.config.degradation_threshold
        relative_change = abs(current - reference) / reference
        
        if metric in ['log_loss', 'brier']:
            return current > reference * (1 + threshold)
        else:
            return current < reference * (1 - threshold)
    
    def _should_retrain(
        self,
        metrics: Dict[str, PerformanceMetrics]
    ) -> bool:
        """Determine if model retraining is recommended"""
        if not self.last_retrain_time:
            return False
        
        # Check if minimum time between retraining has passed
        min_retrain_interval = timedelta(days=self.config.min_retrain_days or 7)
        if datetime.now() - self.last_retrain_time < min_retrain_interval:
            return False
        
        # Count degraded critical metrics
        n_degraded = sum(
            1 for m in metrics.values()
            if m.degraded and m.trend == 'degrading'
        )
        
        return n_degraded >= self.config.retrain_threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self._initialized:
            raise ValueError("Monitor not initialized")
        
        return {
            'current_metrics': {
                metric: list(history)[-1][1]
                for metric, history in self.metric_history.items()
            },
            'trends': {
                metric: {
                    'slope': np.polyfit(
                        [i for i in range(len(history))],
                        [v for _, v in history],
                        1
                    )[0]
                    for metric, history in self.metric_history.items()
                }
            },
            'feature_importance': self.feature_importance,
            'last_retrain_time': self.last_retrain_time,
            'degraded_metrics': [
                metric for metric, history in self.metric_history.items()
                if self._is_metric_degraded(
                    metric,
                    list(history)[-1][1],
                    self.reference_metrics[metric]
                )
            ]
        }
