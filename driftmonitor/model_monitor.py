import os
import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime

class ModelMonitor:
    """A class for monitoring machine learning model performance over time."""
    
    SUPPORTED_METRICS = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score
    }
    
    def __init__(
        self,
        model: Any,
        metrics: Union[str, List[str]] = "accuracy",
        threshold: float = 0.1,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the model monitor.
        
        Args:
            model: The machine learning model to monitor
            metrics: Metric or list of metrics to track (from SUPPORTED_METRICS)
            threshold: Performance degradation threshold to trigger alerts
            save_dir: Directory to save monitoring results (optional)
        """
        self.model = model
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.threshold = threshold
        self.save_dir = save_dir
        
        for metric in self.metrics:
            if metric not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: {metric}. Choose from {list(self.SUPPORTED_METRICS.keys())}")
        
        self.performance_history = []
        self.baseline_performance = None
    
    def track_performance(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Track the model's performance on new data.
        
        Args:
            data: Input features
            labels: True labels
            timestamp: Optional timestamp for the measurement
            
        Returns:
            Dictionary containing computed metric values
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        predictions = self.model.predict(data)
        metrics_values = {}
        
        for metric in self.metrics:
            metric_func = self.SUPPORTED_METRICS[metric]
            try:
                if metric == "roc_auc":
                    prob_predictions = self.model.predict_proba(data)[:, 1]
                    value = metric_func(labels, prob_predictions)
                else:
                    value = metric_func(labels, predictions)
                metrics_values[metric] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric}: {str(e)}")
                metrics_values[metric] = None
        
        performance_record = {
            "timestamp": timestamp,
            "metrics": metrics_values,
            "sample_size": len(labels)
        }
        self.performance_history.append(performance_record)
        
        if self.baseline_performance is None:
            self.baseline_performance = metrics_values
        
        self._check_degradation(metrics_values)
        
        if self.save_dir:
            self._save_results()
            
        return metrics_values
    
    def _check_degradation(self, current_metrics: Dict[str, float]) -> None:
        """Check if performance has degraded beyond the threshold."""
        if self.baseline_performance is None:
            return
            
        for metric in self.metrics:
            baseline = self.baseline_performance[metric]
            current = current_metrics[metric]
            
            if baseline is not None and current is not None:
                degradation = baseline - current
                if degradation > self.threshold:
                    print(f"Alert: {metric} has degraded by {degradation:.3f} "
                          f"(baseline: {baseline:.3f}, current: {current:.3f})")
    
    def _save_results(self) -> None:
        """Save monitoring results to disk."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        filepath = os.path.join(self.save_dir, "monitoring_history.json")
        
        history_serializable = []
        for record in self.performance_history:
            record_copy = record.copy()
            record_copy["timestamp"] = record_copy["timestamp"].isoformat()
            history_serializable.append(record_copy)
            
        with open(filepath, "w") as f:
            json.dump(history_serializable, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model's performance history."""
        if not self.performance_history:
            return {"status": "No performance data available"}
            
        summary = {
            "n_measurements": len(self.performance_history),
            "time_span": {
                "start": self.performance_history[0]["timestamp"],
                "end": self.performance_history[-1]["timestamp"]
            },
            "metrics": {}
        }
        
        for metric in self.metrics:
            values = [record["metrics"][metric] 
                     for record in self.performance_history 
                     if record["metrics"][metric] is not None]
            
            if values:
                summary["metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1]
                }
                
        return summary