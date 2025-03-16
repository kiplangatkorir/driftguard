"""
Model monitoring module for DriftGuard.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, max_error
)

from .interfaces import IModelMonitor
from .config import MonitorConfig

class ModelMonitor(IModelMonitor):
    """Monitors model performance metrics"""
    
    def __init__(self, config: MonitorConfig):
        """Initialize model monitor"""
        self.config = config
        self.model_type = None
        self.metrics_history = pd.DataFrame()
        self._initialized = False
    
    def initialize(self, model_type: str) -> None:
        """Initialize monitor for specific model type"""
        if model_type not in ["classification", "regression"]:
            raise ValueError(
                "Model type must be either 'classification' or 'regression'"
            )
        
        self.model_type = model_type
        self._initialized = True
    
    def _compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """Compute classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle binary classification
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            metrics['precision'] = float(
                precision_score(y_true, y_pred, average='binary')
            )
            metrics['recall'] = float(
                recall_score(y_true, y_pred, average='binary')
            )
            metrics['f1'] = float(
                f1_score(y_true, y_pred, average='binary')
            )
            
            # ROC AUC requires probability scores
            try:
                metrics['roc_auc'] = float(
                    roc_auc_score(y_true, y_pred)
                )
            except Exception:
                pass
        else:
            # Multi-class metrics
            metrics['precision'] = float(
                precision_score(y_true, y_pred, average='weighted')
            )
            metrics['recall'] = float(
                recall_score(y_true, y_pred, average='weighted')
            )
            metrics['f1'] = float(
                f1_score(y_true, y_pred, average='weighted')
            )
        
        return metrics
    
    def _compute_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """Compute regression metrics"""
        metrics = {}
        
        # Error metrics
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['max_error'] = float(max_error(y_true, y_pred))
        
        # Goodness of fit metrics
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(
            explained_variance_score(y_true, y_pred)
        )
        
        return metrics
    
    def track_performance(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Track model performance"""
        if not self._initialized:
            raise ValueError("Monitor not initialized")
        
        if len(predictions) != len(actuals):
            raise ValueError(
                "Length mismatch between predictions and actuals"
            )
        
        # Compute metrics based on model type
        if self.model_type == "classification":
            metrics = self._compute_classification_metrics(
                actuals, predictions
            )
        else:
            metrics = self._compute_regression_metrics(
                actuals, predictions
            )
        
        # Add timestamp
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update metrics history
        metrics['timestamp'] = timestamp
        self.metrics_history = pd.concat([
            self.metrics_history,
            pd.DataFrame([metrics])
        ], ignore_index=True)
        
        return metrics
    
    def check_degradation(
        self,
        metric: str,
        window: Optional[int] = None
    ) -> bool:
        """Check for performance degradation"""
        if metric not in self.metrics_history.columns:
            raise ValueError(f"Unknown metric: {metric}")
        
        if len(self.metrics_history) < 2:
            return False
        
        # Get metric history
        metric_history = self.metrics_history[metric]
        
        if window:
            metric_history = metric_history.tail(window)
        
        if len(metric_history) < 2:
            return False
        
        # Compute baseline and current performance
        baseline = metric_history.iloc[0]
        current = metric_history.iloc[-1]
        
        # Check for degradation
        if self.model_type == "classification":
            # For classification metrics, lower is worse
            degradation = (baseline - current) / baseline
        else:
            # For regression error metrics, higher is worse
            if metric in ['mse', 'rmse', 'mae', 'max_error']:
                degradation = (current - baseline) / baseline
            else:
                # For R2 and explained variance, lower is worse
                degradation = (baseline - current) / baseline
        
        return degradation > self.config.degradation_threshold
    
    def get_performance_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get performance metrics history"""
        metrics = self.metrics_history.copy()
        
        if start_time:
            metrics = metrics[metrics['timestamp'] >= start_time]
        if end_time:
            metrics = metrics[metrics['timestamp'] <= end_time]
        
        return metrics.sort_values('timestamp')
