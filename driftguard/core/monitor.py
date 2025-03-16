"""
Advanced model monitoring system for DriftGuard.
Supports multiple model types and metrics with performance tracking.
"""
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn import metrics
import warnings
from .interfaces import IModelMonitor
from .config import MonitoringConfig, MetricConfig
from .state import StateManager

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    timestamp: datetime
    metrics: Dict[str, float]
    sample_size: int
    prediction_time: float
    metadata: Optional[Dict[str, Any]] = None

class MetricCalculator:
    """Handles calculation of various model metrics"""
    
    CLASSIFICATION_METRICS = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "roc_auc": metrics.roc_auc_score,
        "log_loss": metrics.log_loss,
    }
    
    REGRESSION_METRICS = {
        "mse": metrics.mean_squared_error,
        "rmse": lambda y, yp: np.sqrt(metrics.mean_squared_error(y, yp)),
        "mae": metrics.mean_absolute_error,
        "r2": metrics.r2_score,
        "mape": lambda y, yp: np.mean(np.abs((y - yp) / (y + 1e-10))) * 100,
    }
    
    @staticmethod
    def get_metric_func(
        metric_name: str,
        model_type: str = "classification"
    ) -> Callable:
        """Get metric calculation function"""
        if model_type == "classification":
            if metric_name not in MetricCalculator.CLASSIFICATION_METRICS:
                raise ValueError(f"Unknown classification metric: {metric_name}")
            return MetricCalculator.CLASSIFICATION_METRICS[metric_name]
        else:
            if metric_name not in MetricCalculator.REGRESSION_METRICS:
                raise ValueError(f"Unknown regression metric: {metric_name}")
            return MetricCalculator.REGRESSION_METRICS[metric_name]
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics_list: List[str],
        model_type: str = "classification"
    ) -> Dict[str, float]:
        """Calculate multiple metrics"""
        results = {}
        
        for metric_name in metrics_list:
            try:
                metric_func = MetricCalculator.get_metric_func(metric_name, model_type)
                if metric_name == "roc_auc":
                    # Special handling for ROC AUC which needs probabilities
                    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                        y_pred_prob = y_pred[:, 1]
                    else:
                        y_pred_prob = y_pred
                    results[metric_name] = float(metric_func(y_true, y_pred_prob))
                else:
                    results[metric_name] = float(metric_func(y_true, y_pred))
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
                results[metric_name] = None
                
        return results

class BaseModelMonitor(IModelMonitor):
    """Base class for model monitoring"""
    
    def __init__(
        self,
        config: MonitoringConfig,
        state_manager: StateManager,
        model_type: str = "classification"
    ):
        self.config = config
        self.state_manager = state_manager
        self.model_type = model_type
        self.metric_calculator = MetricCalculator()
        self.baseline_metrics = None
        
    def track_performance(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Track model performance metrics"""
        start_time = datetime.now()
        
        # Calculate metrics
        metrics_list = [m.name for m in self.config.metrics]
        performance_metrics = self.metric_calculator.calculate_metrics(
            actual,
            predictions,
            metrics_list,
            self.model_type
        )
        
        # Record prediction time
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Create metrics record
        metrics_record = ModelMetrics(
            timestamp=datetime.now(),
            metrics=performance_metrics,
            sample_size=len(features),
            prediction_time=prediction_time,
            metadata=metadata
        )
        
        # Update state
        self._update_state(metrics_record)
        
        # Check for performance degradation
        self._check_performance(metrics_record)
        
        return performance_metrics
    
    def _update_state(self, metrics: ModelMetrics) -> None:
        """Update monitoring state"""
        try:
            # Update metrics history
            self.state_manager.update_metrics(
                metrics.metrics,
                metrics.timestamp
            )
            
            # Update baseline if not set
            if self.baseline_metrics is None:
                self.baseline_metrics = metrics.metrics
                self.state_manager.save_state({
                    "baseline_metrics": self.baseline_metrics
                })
        except Exception as e:
            logger.error(f"Failed to update state: {str(e)}")
    
    def _check_performance(self, metrics: ModelMetrics) -> None:
        """Check for performance degradation"""
        if self.baseline_metrics is None:
            return
            
        degradation_detected = False
        degraded_metrics = []
        
        for metric_config in self.config.metrics:
            current_value = metrics.metrics.get(metric_config.name)
            baseline_value = self.baseline_metrics.get(metric_config.name)
            
            if current_value is None or baseline_value is None:
                continue
                
            # Check if performance has degraded beyond threshold
            if metric_config.name in self.metric_calculator.CLASSIFICATION_METRICS:
                degradation = baseline_value - current_value
            else:  # For error metrics, higher is worse
                degradation = current_value - baseline_value
                
            if degradation > metric_config.threshold:
                degradation_detected = True
                degraded_metrics.append({
                    "metric": metric_config.name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation": degradation
                })
        
        if degradation_detected:
            self.state_manager.add_warning(
                f"Performance degradation detected in metrics: "
                f"{', '.join(d['metric'] for d in degraded_metrics)}"
            )
    
    def get_performance_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get historical performance data"""
        return self.state_manager.get_metrics_history(start_time, end_time)
    
    def get_performance_summary(
        self,
        window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if window is None:
            window = timedelta(days=7)
            
        end_time = datetime.now()
        start_time = end_time - window
        
        metrics_df = self.get_performance_history(start_time, end_time)
        if metrics_df.empty:
            return {}
            
        summary = {
            "time_window": {
                "start": start_time,
                "end": end_time
            },
            "sample_size": len(metrics_df),
            "metrics": {}
        }
        
        # Calculate summary statistics for each metric
        metric_columns = [m.name for m in self.config.metrics]
        for metric in metric_columns:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if not values.empty:
                    summary["metrics"][metric] = {
                        "current": float(values.iloc[-1]),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "trend": self._calculate_trend(values)
                    }
        
        return summary
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction using linear regression"""
        if len(values) < 2:
            return "stable"
            
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        return "improving" if slope > 0 else "degrading"
    
    def set_baseline(
        self,
        metrics: Optional[Dict[str, float]] = None,
        window: Optional[timedelta] = None
    ) -> None:
        """Set or update baseline metrics"""
        if metrics is not None:
            self.baseline_metrics = metrics
        elif window is not None:
            # Use average over window as baseline
            end_time = datetime.now()
            start_time = end_time - window
            metrics_df = self.get_performance_history(start_time, end_time)
            
            if not metrics_df.empty:
                self.baseline_metrics = metrics_df.mean().to_dict()
        else:
            raise ValueError("Either metrics or window must be provided")
            
        self.state_manager.save_state({
            "baseline_metrics": self.baseline_metrics
        })

class ClassificationModelMonitor(BaseModelMonitor):
    """Monitor for classification models"""
    
    def __init__(
        self,
        config: MonitoringConfig,
        state_manager: StateManager
    ):
        super().__init__(config, state_manager, "classification")
        
    def track_performance(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Track classification model performance"""
        # Add classification-specific metadata
        if metadata is None:
            metadata = {}
            
        # Calculate class distribution
        class_dist = pd.Series(actual).value_counts(normalize=True).to_dict()
        metadata["class_distribution"] = class_dist
        
        # Calculate confusion matrix
        try:
            conf_matrix = metrics.confusion_matrix(actual, predictions)
            metadata["confusion_matrix"] = conf_matrix.tolist()
        except Exception as e:
            logger.warning(f"Failed to calculate confusion matrix: {str(e)}")
        
        return super().track_performance(features, predictions, actual, metadata)

class RegressionModelMonitor(BaseModelMonitor):
    """Monitor for regression models"""
    
    def __init__(
        self,
        config: MonitoringConfig,
        state_manager: StateManager
    ):
        super().__init__(config, state_manager, "regression")
        
    def track_performance(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Track regression model performance"""
        # Add regression-specific metadata
        if metadata is None:
            metadata = {}
            
        # Calculate prediction distribution statistics
        metadata["prediction_stats"] = {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "min": float(predictions.min()),
            "max": float(predictions.max())
        }
        
        # Calculate residuals
        residuals = actual - predictions
        metadata["residual_stats"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "skew": float(np.mean(residuals**3) / (np.std(residuals)**3)),
            "kurtosis": float(np.mean(residuals**4) / (np.std(residuals)**4))
        }
        
        return super().track_performance(features, predictions, actual, metadata)

def create_model_monitor(
    model_type: str,
    config: MonitoringConfig,
    state_manager: StateManager
) -> BaseModelMonitor:
    """Factory function to create model monitor"""
    if model_type == "classification":
        return ClassificationModelMonitor(config, state_manager)
    elif model_type == "regression":
        return RegressionModelMonitor(config, state_manager)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class ModelMonitor(IModelMonitor):
    """Model performance monitoring"""
    
    def __init__(
        self,
        model_type: str = "classification",
        metrics: Optional[List[str]] = None,
        window_size: int = 1000
    ):
        """Initialize monitor"""
        self.model_type = model_type
        self.metrics = metrics or self._default_metrics()
        self.window_size = window_size
        self.model = None
        self.reference_data = None
        self.reference_performance = {}
        
        # Validate model type
        if model_type not in ["classification", "regression"]:
            raise ValueError(
                "Model type must be 'classification' or 'regression'"
            )
    
    def _default_metrics(self) -> List[str]:
        """Get default metrics based on model type"""
        if self.model_type == "classification":
            return ["accuracy", "f1", "roc_auc"]
        return ["mse", "rmse", "mae"]
    
    def initialize(
        self,
        model: Any,
        reference_data: pd.DataFrame
    ) -> None:
        """Initialize monitor with model and reference data"""
        try:
            self.model = model
            self.reference_data = reference_data.copy()
            logger.info("Initialized model monitor")
        except Exception as e:
            logger.error(f"Failed to initialize monitor: {str(e)}")
            raise
    
    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute classification metrics"""
        results = {}
        
        if "accuracy" in self.metrics:
            results["accuracy"] = metrics.accuracy_score(y_true, y_pred)
        
        if "precision" in self.metrics:
            results["precision"] = metrics.precision_score(
                y_true, y_pred, average='weighted'
            )
        
        if "recall" in self.metrics:
            results["recall"] = metrics.recall_score(
                y_true, y_pred, average='weighted'
            )
        
        if "f1" in self.metrics:
            results["f1"] = metrics.f1_score(
                y_true, y_pred, average='weighted'
            )
        
        if "roc_auc" in self.metrics and y_prob is not None:
            # For multiclass, compute one-vs-rest ROC AUC
            results["roc_auc"] = metrics.roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr'
            )
        
        return results
    
    def _compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics"""
        results = {}
        
        if "mse" in self.metrics:
            results["mse"] = metrics.mean_squared_error(y_true, y_pred)
        
        if "rmse" in self.metrics:
            results["rmse"] = np.sqrt(
                metrics.mean_squared_error(y_true, y_pred)
            )
        
        if "mae" in self.metrics:
            results["mae"] = metrics.mean_absolute_error(y_true, y_pred)
        
        return results
    
    def track_performance(
        self,
        data: pd.DataFrame,
        actual_labels: Union[pd.Series, np.ndarray]
    ) -> List[MetricReport]:
        """Track model performance"""
        if self.model is None:
            raise ValueError("Monitor not initialized")
        
        try:
            # Get predictions
            y_pred = self.model.predict(data)
            y_prob = None
            if hasattr(self.model, "predict_proba"):
                y_prob = self.model.predict_proba(data)
            
            # Compute metrics
            if self.model_type == "classification":
                results = self._compute_classification_metrics(
                    actual_labels, y_pred, y_prob
                )
            else:
                results = self._compute_regression_metrics(
                    actual_labels, y_pred
                )
            
            # Create metric reports
            reports = []
            for metric_name, value in results.items():
                reports.append(MetricReport(
                    metric_name=metric_name,
                    value=float(value),
                    threshold=0.1,  # Default threshold
                    exceeds_threshold=False,  # Will be set by monitor
                    timestamp=datetime.now()
                ))
            
            return reports
            
        except Exception as e:
            logger.error(f"Failed to track performance: {str(e)}")
            raise
    
    def update_reference(
        self,
        new_reference: pd.DataFrame,
        new_labels: Optional[pd.Series] = None
    ) -> None:
        """Update reference data"""
        try:
            self.reference_data = new_reference.copy()
            
            # Update reference performance if labels provided
            if new_labels is not None:
                self.reference_performance = {}
                reports = self.track_performance(new_reference, new_labels)
                for report in reports:
                    self.reference_performance[report.metric_name] = report.value
            
            logger.info("Updated reference data")
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")
            raise
