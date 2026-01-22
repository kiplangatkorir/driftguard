"""
Model monitoring module for DriftGuard.
"""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from sklearn import metrics, clone
from datetime import datetime
import logging

if TYPE_CHECKING:
    from ..alert_manager import AlertManager

from .interfaces import IModelMonitor
from .config import MonitorConfig

logger = logging.getLogger(__name__)

class ModelMonitor(IModelMonitor):
    """Monitors model performance and detects concept drift"""
    
    def __init__(self, config: Optional[MonitorConfig] = None, alert_manager: Optional['AlertManager'] = None):
        """Initialize model monitor"""
        self.config = config or MonitorConfig()
        self.reference_metrics = {}
        self.reference_predictions = None
        self.reference_labels = None
        self.model = None  # Track the current model
        self.retrain_count = 0
        self._initialized = False
        self.alert_manager = alert_manager
    
    def initialize(
        self,
        reference_predictions: pd.Series,
        reference_labels: pd.Series
    ) -> None:
        """Initialize monitor with reference data"""
        if len(reference_predictions) != len(reference_labels):
            raise ValueError(
                "Length mismatch between predictions and labels"
            )
        
        if len(reference_predictions) == 0:
            raise ValueError("Reference data cannot be empty")
        
        self.reference_predictions = reference_predictions.copy()
        self.reference_labels = reference_labels.copy()
        
        # Compute reference metrics
        self.reference_metrics = self._compute_metrics(
            reference_predictions,
            reference_labels
        )
        
        self._initialized = True
    
    def attach_model(self, model):
        """Attach a model for potential retraining"""
        self.model = model
    
    def attach_alert_manager(self, alert_manager: 'AlertManager') -> None:
        """Attach an AlertManager for performance alerts"""
        self.alert_manager = alert_manager
        
    def should_retrain(self, current_performance: dict) -> bool:
        """Determine if retraining is needed based on performance drop"""
        if not self.model:
            return False
            
        baseline_acc = self.reference_metrics['accuracy']
        current_acc = current_performance['accuracy']['value']
        return (baseline_acc - current_acc) > self.config.retrain_threshold
        
    def retrain_model(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Retrain the attached model with combined data"""
        if not hasattr(self.model, 'fit'):
            raise ValueError("Attached model does not support retraining")
            
        logger.info(f"Retraining model (attempt {self.retrain_count + 1})")
        
        # Clone and retrain model
        new_model = clone(self.model)
        new_model.fit(X_new, y_new)
        
        self.retrain_count += 1
        return new_model
    
    def track(
        self,
        predictions: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Track model performance"""
        if not self._initialized:
            raise ValueError("Monitor not initialized")
        
        if len(predictions) != len(labels):
            raise ValueError(
                "Length mismatch between predictions and labels"
            )
        
        # Compute current metrics
        current_metrics = self._compute_metrics(predictions, labels)
        
        # Check for degradation
        degraded_metrics = self._check_degradation(current_metrics)
        
        # Add degradation flags to metrics
        metrics_with_status = {
            metric: {
                'value': value,
                'degraded': metric in degraded_metrics,
                'reference': self.reference_metrics[metric]
            }
            for metric, value in current_metrics.items()
        }
        
        # Check if retraining is needed
        if self.should_retrain(metrics_with_status):
            # Retrain the model
            new_model = self.retrain_model(
                pd.concat([self.reference_labels, labels]),
                pd.concat([self.reference_predictions, predictions])
            )
            # Update the model
            self.model = new_model
        
        return metrics_with_status
    
    def _compute_metrics(
        self,
        predictions: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Compute performance metrics"""
        metrics_dict = {}
        
        # Handle binary classification metrics
        if len(np.unique(labels)) == 2:
            if 'accuracy' in self.config.metrics:
                metrics_dict['accuracy'] = metrics.accuracy_score(
                    labels,
                    predictions
                )
            
            if 'precision' in self.config.metrics:
                metrics_dict['precision'] = metrics.precision_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            
            if 'recall' in self.config.metrics:
                metrics_dict['recall'] = metrics.recall_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            
            if 'f1' in self.config.metrics:
                metrics_dict['f1'] = metrics.f1_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            
            if 'roc_auc' in self.config.metrics:
                try:
                    metrics_dict['roc_auc'] = metrics.roc_auc_score(
                        labels,
                        predictions
                    )
                except:
                    metrics_dict['roc_auc'] = 0.5
        
        # Handle multiclass classification metrics
        else:
            if 'accuracy' in self.config.metrics:
                metrics_dict['accuracy'] = metrics.accuracy_score(
                    labels,
                    predictions
                )
            
            if 'precision' in self.config.metrics:
                metrics_dict['precision'] = metrics.precision_score(
                    labels,
                    predictions,
                    average='weighted',
                    zero_division=0
                )
            
            if 'recall' in self.config.metrics:
                metrics_dict['recall'] = metrics.recall_score(
                    labels,
                    predictions,
                    average='weighted',
                    zero_division=0
                )
            
            if 'f1' in self.config.metrics:
                metrics_dict['f1'] = metrics.f1_score(
                    labels,
                    predictions,
                    average='weighted',
                    zero_division=0
                )
        
        return metrics_dict
    
    def _check_degradation(
        self,
        current_metrics: Dict[str, float]
    ) -> List[str]:
        """Check for performance degradation and trigger alerts"""
        degraded_metrics = []
        
        for metric, value in current_metrics.items():
            reference_value = self.reference_metrics[metric]
            threshold = self.config.thresholds[metric]
            
            degraded = False
            degradation_pct = 0.0
            
            if self.config.threshold_type == 'absolute':
                if value < threshold:
                    degraded = True
                    degradation_pct = ((threshold - value) / threshold) * 100
            
            elif self.config.threshold_type == 'relative':
                relative_change = (value - reference_value) / reference_value
                if relative_change < -threshold:
                    degraded = True
                    degradation_pct = abs(relative_change) * 100
            
            elif self.config.threshold_type == 'dynamic':
                # Use statistical process control
                if self._detect_significant_change(
                    metric,
                    value,
                    reference_value,
                    threshold
                ):
                    degraded = True
                    degradation_pct = abs((value - reference_value) / reference_value) * 100
            
            if degraded:
                degraded_metrics.append(metric)
                
                # Trigger alert if AlertManager is attached
                if self.alert_manager:
                    alert_message = f"""Model Performance Degradation Detected

Metric: {metric}
Baseline Value: {reference_value:.4f}
Current Value: {value:.4f}
Degradation: {degradation_pct:.2f}%
Threshold: {threshold}

This metric has degraded beyond the acceptable threshold."""
                    # Calculate drift score for alert threshold
                    drift_score = degradation_pct / 100
                    self.alert_manager.check_and_alert(
                        drift_score=drift_score,
                        message=alert_message
                    )
        
        return degraded_metrics
    
    def _detect_significant_change(
        self,
        metric: str,
        current_value: float,
        reference_value: float,
        threshold: float
    ) -> bool:
        """Detect statistically significant performance change"""
        # Implement statistical process control
        # Using 3-sigma rule for significant changes
        std_dev = threshold * reference_value
        lower_bound = reference_value - 3 * std_dev
        
        return current_value < lower_bound
    
    def detect_concept_drift(
        self,
        predictions: pd.Series,
        labels: pd.Series,
        window_size: Optional[int] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect concept drift using performance metrics"""
        if not self._initialized:
            raise ValueError("Monitor not initialized")
        
        window_size = window_size or self.config.window_size
        
        # Get current window metrics
        current_metrics = self._compute_metrics(
            predictions[-window_size:],
            labels[-window_size:]
        )
        
        # Check for significant degradation
        degraded_metrics = self._check_degradation(current_metrics)
        
        # Compute drift severity
        drift_metrics = {}
        for metric in self.config.metrics:
            if metric in current_metrics:
                reference_value = self.reference_metrics[metric]
                current_value = current_metrics[metric]
                
                # Compute relative change
                if reference_value != 0:
                    relative_change = (
                        current_value - reference_value
                    ) / reference_value
                else:
                    relative_change = float('inf')
                
                drift_metrics[metric] = {
                    'current': current_value,
                    'reference': reference_value,
                    'relative_change': relative_change,
                    'degraded': metric in degraded_metrics
                }
        
        has_drift = len(degraded_metrics) > 0
        
        return has_drift, drift_metrics
