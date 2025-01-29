import logging
from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
from driftmonitor.drift_detector import DriftDetector
from driftmonitor.model_monitor import ModelMonitor
from driftmonitor.alert_manager import AlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DriftMonitor')

class DriftMonitorWrapper:
    def __init__(
        self,
        model,
        reference_data: pd.DataFrame,
        alert_email: Optional[str] = None,
        alert_threshold: float = 0.5,
        monitor_name: str = "Model Monitor"
    ):
        """
        Initialize drift monitoring with minimal configuration.
        
        Args:
            model: The trained model to monitor
            reference_data: Training/reference data for drift comparison
            alert_email: Email to receive drift alerts (optional)
            alert_threshold: Threshold for drift alerts (default: 0.5)
            monitor_name: Name for this monitoring instance
        """
        self.model = model
        self.reference_data = reference_data
        self.monitor_name = monitor_name
        
        self.model_monitor = ModelMonitor(model)
        self.drift_detector = DriftDetector(reference_data)
        self.alert_manager = AlertManager(threshold=alert_threshold)
        
        if alert_email:
            try:
                self.alert_manager.set_recipient_email(
                    alert_email,
                    monitor_name
                )
                logger.info(f"Alerts will be sent to {alert_email}")
            except ValueError as e:
                logger.warning(f"Invalid email configuration: {e}")

    def monitor(
        self,
        new_data: pd.DataFrame,
        actual_labels: Optional[Union[pd.Series, np.ndarray]] = None,
        raise_on_drift: bool = False
    ) -> Dict:
        """
        Monitor new data for drift and performance degradation.
        
        Args:
            new_data: New data to monitor
            actual_labels: True labels if available (for performance monitoring)
            raise_on_drift: Whether to raise an exception on detected drift
            
        Returns:
            Dict containing monitoring results
        """
        results = {
            'has_drift': False,
            'drift_detected_in': [],
            'performance': None,
            'drift_scores': {}
        }
        
        drift_report = self.drift_detector.detect_drift(new_data)
        
        for feature, report in drift_report.items():
            drift_score = report['drift_score']
            results['drift_scores'][feature] = drift_score
            
            if drift_score > self.alert_manager.threshold:
                results['has_drift'] = True
                results['drift_detected_in'].append(feature)
                
                message = (
                    f"Drift detected in feature '{feature}'\n"
                    f"Drift Score: {drift_score:.3f}\n"
                    f"P-value: {report['p_value']:.3f}"
                )
                self.alert_manager.check_and_alert(drift_score, message)
        
        if actual_labels is not None:
            performance = self.model_monitor.track_performance(new_data, actual_labels)
            results['performance'] = performance
        
        if results['has_drift']:
            logger.warning(
                f"Drift detected in {len(results['drift_detected_in'])} features: "
                f"{', '.join(results['drift_detected_in'])}"
            )
            if raise_on_drift:
                raise ValueError("Data drift detected above threshold")
        else:
            logger.info("No significant drift detected")
            
        return results

    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics."""
        return {
            'alerts': self.alert_manager.get_alert_statistics(),
            'performance_history': self.model_monitor.performance_history
        }