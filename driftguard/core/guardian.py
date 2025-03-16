"""
Main DriftGuard class that integrates all monitoring components.
"""
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .interfaces import (
    IDriftDetector,
    IModelMonitor,
    IStateManager,
    IDataValidator,
    DriftReport,
    MetricReport
)
from .config import DriftConfig, ModelConfig
from .drift import create_drift_detector
from .monitor import ModelMonitor
from .state import StateManager
from .validation import DataValidator

logger = logging.getLogger(__name__)

class DriftGuard:
    """Main class for model monitoring and drift detection"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        drift_config: DriftConfig,
        storage_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize DriftGuard"""
        self.model_config = model_config
        self.drift_config = drift_config
        self.storage_config = storage_config or {'path': '.driftguard'}
        
        # Initialize components
        self.drift_detector = create_drift_detector(
            drift_config.method,
            drift_config
        )
        
        self.model_monitor = ModelMonitor(
            model_type=model_config.type,
            metrics=model_config.metrics
        )
        
        self.state_manager = StateManager(self.storage_config)
        
        self.validator = DataValidator(
            max_missing_pct=model_config.max_missing_pct
        )
        
        self._initialized = False
    
    def initialize(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        reference_labels: Optional[pd.Series] = None
    ) -> None:
        """Initialize monitoring with reference data"""
        try:
            # Validate reference data
            validation_result = self.validator.validate(reference_data)
            if not validation_result.is_valid:
                raise ValueError(
                    "Reference data validation failed:\n" +
                    "\n".join(validation_result.errors)
                )
            
            # Initialize components
            self.validator.initialize(reference_data)
            self.drift_detector.initialize(reference_data)
            self.model_monitor.initialize(model, reference_data)
            
            # Update state
            self.state_manager.save_state({
                'last_update': datetime.now().isoformat(),
                'monitoring': {
                    'total_samples_processed': 0,
                    'last_drift_detected': None,
                    'reference_data_shape': reference_data.shape
                }
            })
            
            self._initialized = True
            logger.info("Initialized DriftGuard")
            
        except Exception as e:
            logger.error(f"Failed to initialize DriftGuard: {str(e)}")
            raise
    
    def monitor(
        self,
        data: pd.DataFrame,
        actual_labels: Optional[pd.Series] = None,
        raise_on_drift: bool = False
    ) -> Dict[str, Any]:
        """Monitor new data for drift and performance"""
        if not self._initialized:
            raise ValueError("DriftGuard not initialized")
        
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': data.shape,
                'drift_detected': False,
                'validation_passed': False,
                'drift_reports': [],
                'metric_reports': [],
                'validation_result': None
            }
            
            # Validate input data
            validation_result = self.validator.validate(data)
            results['validation_result'] = validation_result.dict()
            
            if not validation_result.is_valid:
                if raise_on_drift:
                    raise ValueError(
                        "Data validation failed:\n" +
                        "\n".join(validation_result.errors)
                    )
                return results
            
            results['validation_passed'] = True
            
            # Check for drift
            drift_reports = self.drift_detector.detect_drift(data)
            results['drift_reports'] = [
                report.dict() for report in drift_reports
            ]
            
            # Track performance if labels provided
            if actual_labels is not None:
                metric_reports = self.model_monitor.track_performance(
                    data, actual_labels
                )
                results['metric_reports'] = [
                    report.dict() for report in metric_reports
                ]
            
            # Check if drift detected
            drift_detected = any(
                report.has_drift for report in drift_reports
            )
            results['drift_detected'] = drift_detected
            
            # Update state
            state = self.state_manager.load_state()
            state['monitoring']['total_samples_processed'] += len(data)
            if drift_detected:
                state['monitoring']['last_drift_detected'] = datetime.now().isoformat()
            self.state_manager.save_state(state)
            
            # Update metrics history
            metrics = {}
            if actual_labels is not None:
                for report in metric_reports:
                    metrics[report.metric_name] = report.value
            for report in drift_reports:
                metrics[f"drift_{report.feature_name}"] = report.drift_score
            
            self.state_manager.update_metrics(metrics)
            
            if drift_detected and raise_on_drift:
                raise ValueError(
                    "Drift detected in features: " +
                    ", ".join(
                        report.feature_name
                        for report in drift_reports
                        if report.has_drift
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            raise
    
    def update_reference(
        self,
        new_reference: pd.DataFrame,
        new_labels: Optional[pd.Series] = None
    ) -> None:
        """Update reference data"""
        if not self._initialized:
            raise ValueError("DriftGuard not initialized")
        
        try:
            # Validate new reference data
            validation_result = self.validator.validate(new_reference)
            if not validation_result.is_valid:
                raise ValueError(
                    "New reference data validation failed:\n" +
                    "\n".join(validation_result.errors)
                )
            
            # Update components
            self.validator.update_reference(new_reference)
            self.drift_detector.update_reference(new_reference)
            self.model_monitor.update_reference(new_reference, new_labels)
            
            # Update state
            state = self.state_manager.load_state()
            state['last_update'] = datetime.now().isoformat()
            state['monitoring']['reference_data_shape'] = new_reference.shape
            self.state_manager.save_state(state)
            
            logger.info("Updated reference data")
            
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        if not self._initialized:
            raise ValueError("DriftGuard not initialized")
        
        try:
            return self.state_manager.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get status: {str(e)}")
            raise
