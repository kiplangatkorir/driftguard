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
from .config import ModelConfig, DriftConfig
from .drift import KSTestDriftDetector, JSDDriftDetector, PSIDriftDetector
from .monitor import ModelMonitor
from .validation import DataValidator
from .state import StateManager

logger = logging.getLogger(__name__)

class DriftGuard:
    """
    Main class for model monitoring and drift detection.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        drift_config: DriftConfig,
        storage_config: Dict[str, Any]
    ):
        """Initialize DriftGuard"""
        self.model_config = model_config
        self.drift_config = drift_config
        self.storage_config = storage_config
        
        # Initialize components
        self.drift_detector = self._create_drift_detector()
        self.model_monitor = ModelMonitor(
            model_type=model_config.type,
            metrics=model_config.metrics
        )
        self.state_manager = StateManager(**storage_config)
        self.data_validator = DataValidator(
            max_missing_pct=model_config.max_missing_pct
        )
        
        self._initialized = False
    
    def _create_drift_detector(self) -> IDriftDetector:
        """Create drift detector based on configuration"""
        detectors = {
            'ks_test': KSTestDriftDetector,
            'jsd': JSDDriftDetector,
            'psi': PSIDriftDetector
        }
        
        detector_cls = detectors.get(self.drift_config.method)
        if not detector_cls:
            raise ValueError(f"Unknown drift detection method: {self.drift_config.method}")
        
        return detector_cls(self.drift_config)
    
    def initialize(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        reference_labels: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """Initialize DriftGuard with reference data"""
        try:
            # Validate reference data
            self.data_validator.initialize(reference_data)
            validation_result = self.data_validator.validate(reference_data)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Reference data validation failed: {validation_result.errors}"
                )
            
            # Initialize components
            self.drift_detector.initialize(reference_data)
            self.model_monitor.initialize(model, reference_data)
            
            if reference_labels is not None:
                self.model_monitor.update_reference(reference_data, reference_labels)
            
            # Save initial state
            self.state_manager.save_state({
                'initialized_at': datetime.now().isoformat(),
                'reference_shape': reference_data.shape,
                'features': list(reference_data.columns)
            })
            
            self._initialized = True
            logger.info("DriftGuard initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DriftGuard: {str(e)}")
            raise
    
    def monitor(
        self,
        new_data: pd.DataFrame,
        actual_labels: Optional[Union[pd.Series, np.ndarray]] = None,
        raise_on_drift: bool = False
    ) -> Dict[str, Any]:
        """
        Monitor new data for drift and track model performance.
        
        Args:
            new_data: New data to monitor
            actual_labels: Actual labels for performance tracking
            raise_on_drift: Whether to raise exception on drift detection
            
        Returns:
            Dictionary containing drift and performance reports
        """
        if not self._initialized:
            raise ValueError("DriftGuard not initialized. Call initialize() first.")
        
        try:
            # Validate input data
            validation_result = self.data_validator.validate(new_data)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Input data validation failed: {validation_result.errors}"
                )
            
            # Detect drift
            drift_reports = self.drift_detector.detect_drift(new_data)
            drift_detected = any(report.has_drift for report in drift_reports)
            
            # Track performance if labels provided
            metric_reports = []
            if actual_labels is not None:
                metric_reports = self.model_monitor.track_performance(
                    new_data,
                    actual_labels
                )
            
            # Update state
            self.state_manager.update_metrics({
                'drift_detected': drift_detected,
                'samples_processed': len(new_data)
            })
            
            results = {
                'drift_detected': drift_detected,
                'drift_reports': [
                    report.model_dump() for report in drift_reports
                ],
                'metric_reports': [
                    report.model_dump() for report in metric_reports
                ],
                'validation_passed': validation_result.is_valid,
                'timestamp': datetime.now().isoformat()
            }
            
            if drift_detected and raise_on_drift:
                raise ValueError("Drift detected in input data")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        try:
            return self.state_manager.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            raise
