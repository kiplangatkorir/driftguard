"""
Main DriftGuard class that integrates all monitoring components.
"""
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from .interfaces import (
    DriftReport, MetricReport,
    IDriftDetector, IStateManager,
    IAlertManager, IModelMonitor
)
from .config import DriftGuardConfig
from .drift import create_drift_detector
from .state import StateManager
from .monitor import ModelMonitor
from .alerts import AlertManager
from .validation import DataValidator

logger = logging.getLogger(__name__)

class DriftGuard:
    """Main class for model monitoring and drift detection"""
    
    def __init__(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        config_path: Optional[str] = None,
        model_type: str = "classification"
    ):
        """Initialize DriftGuard"""
        # Load configuration
        self.config = (
            DriftGuardConfig.from_yaml(config_path)
            if config_path else DriftGuardConfig()
        )
        
        # Initialize components
        self._init_components(model, reference_data, model_type)
        
        logger.info(
            f"Initialized DriftGuard v{self.config.version} "
            f"for project '{self.config.project_name}'"
        )
    
    def _init_components(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        model_type: str
    ) -> None:
        """Initialize monitoring components"""
        try:
            # Initialize state manager
            self.state_manager = StateManager(self.config.storage)
            
            # Initialize drift detector
            self.drift_detector = create_drift_detector(
                self.config.drift.method,
                self.config.drift
            )
            self.drift_detector.initialize(reference_data)
            
            # Initialize model monitor
            self.model_monitor = ModelMonitor(
                model_type=model_type,
                metrics=self.config.monitor.metrics,
                window_size=self.config.monitor.window_size
            )
            self.model_monitor.initialize(
                model=model,
                reference_data=reference_data
            )
            
            # Initialize alert manager
            self.alert_manager = AlertManager(self.config.alerts)
            
            # Initialize data validator
            self.data_validator = DataValidator()
            self.data_validator.initialize(reference_data)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    async def monitor_batch(
        self,
        data: pd.DataFrame,
        actual_labels: Optional[pd.Series] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Monitor a batch of data"""
        try:
            # Validate data
            validation_errors = self.data_validator.validate_schema(data)
            if validation_errors:
                return {
                    'status': 'error',
                    'messages': validation_errors
                }
            
            # Check for drift
            drift_reports = self.drift_detector.detect_drift(data)
            drift_detected = any(report.has_drift for report in drift_reports)
            
            # Track performance if labels provided
            performance_reports = []
            if actual_labels is not None:
                performance_reports = self.model_monitor.track_performance(
                    data,
                    actual_labels=actual_labels
                )
            
            # Update state
            timestamp = datetime.now()
            if performance_reports:
                self.state_manager.update_metrics(
                    {
                        report.metric_name: report.value
                        for report in performance_reports
                    },
                    timestamp
                )
            
            # Send alerts if needed
            if drift_detected:
                await self.alert_manager.send_drift_alert(
                    drift_reports,
                    metadata
                )
            
            if any(report.exceeds_threshold for report in performance_reports):
                await self.alert_manager.send_metric_alert(
                    performance_reports,
                    metadata
                )
            
            # Prepare response
            response = {
                'status': 'success',
                'timestamp': timestamp.isoformat(),
                'drift_detected': drift_detected,
                'drift_reports': [
                    report.model_dump() for report in drift_reports
                ]
            }
            
            if performance_reports:
                response['performance_metrics'] = {
                    report.metric_name: report.value
                    for report in performance_reports
                }
            
            if metadata:
                response['metadata'] = metadata
            
            return response
            
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            return {
                'status': 'error',
                'messages': [str(e)]
            }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        try:
            # Get system status
            status = self.state_manager.get_system_status()
            
            # Get recent metrics
            metrics_df = self.state_manager.get_metrics_history(
                start_time=datetime.now() - pd.Timedelta(days=7)
            )
            
            performance_summary = {}
            if not metrics_df.empty:
                for metric in metrics_df.columns:
                    if metric != 'timestamp':
                        performance_summary[metric] = {
                            'current': float(metrics_df[metric].iloc[-1]),
                            'mean': float(metrics_df[metric].mean()),
                            'std': float(metrics_df[metric].std())
                        }
            
            return {
                'status': status,
                'performance': {
                    'metrics': performance_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {str(e)}")
            return {
                'status': 'error',
                'messages': [str(e)]
            }
    
    def update_reference(
        self,
        new_reference: pd.DataFrame,
        new_labels: Optional[pd.Series] = None
    ) -> None:
        """Update reference data"""
        try:
            # Validate new reference data
            validation_errors = self.data_validator.validate_schema(new_reference)
            if validation_errors:
                raise ValueError(
                    f"Invalid reference data: {validation_errors}"
                )
            
            # Update components
            self.drift_detector.update_reference(new_reference)
            self.model_monitor.update_reference(
                new_reference,
                new_labels
            )
            self.data_validator.initialize(new_reference)
            
            logger.info("Updated reference data")
            
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")
            raise
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        try:
            self.config.update(updates)
            logger.info("Updated configuration")
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise
