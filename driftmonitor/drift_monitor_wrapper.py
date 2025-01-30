import os
import logging
from typing import Optional, Union, Dict, Set
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
        # Validate reference data
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
            
        self.model = model
        self.reference_data = reference_data
        self.reference_columns = set(reference_data.columns)
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

    def _validate_input_data(self, new_data: pd.DataFrame) -> None:
        """
        Validate input data against reference data requirements.
        
        Args:
            new_data: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if new_data.empty:
            raise ValueError("Input data cannot be empty")
            
        new_columns = set(new_data.columns)
        
        # Check for missing required columns
        missing_cols = self.reference_columns - new_columns
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Expected columns: {self.reference_columns}"
            )
            
        # Check for unexpected additional columns
        extra_cols = new_columns - self.reference_columns
        if extra_cols:
            raise ValueError(
                f"Unexpected columns found: {extra_cols}. "
                f"Expected columns: {self.reference_columns}"
            )
            
        # Validate data types (optional, but recommended)
        for col in self.reference_columns:
            if new_data[col].dtype != self.reference_data[col].dtype:
                raise ValueError(
                    f"Column '{col}' has incorrect dtype. "
                    f"Expected {self.reference_data[col].dtype}, "
                    f"got {new_data[col].dtype}"
                )

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
        
        Raises:
            ValueError: If input validation fails or if drift is detected with raise_on_drift=True
        """
        # Validate input data before processing
        self._validate_input_data(new_data)
        
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