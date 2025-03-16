"""
Main DriftGuard monitoring system.
Integrates drift detection, model monitoring, and alerting in a modular way.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .config import (
    ConfigManager, MonitoringConfig, DriftConfig,
    AlertConfig, DriftMethod
)
from .state import StateManager
from .drift import create_drift_detector, BaseDriftDetector
from .monitor import create_model_monitor, BaseModelMonitor
from .alerts import AlertManager, Alert, AlertLevel
from .validation import DataSchema, DataValidator
from .interfaces import DriftReport

logger = logging.getLogger(__name__)

class DriftGuard:
    """
    Main class for model monitoring and drift detection.
    Provides a clean interface while maintaining separation of concerns.
    """
    
    def __init__(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        config_path: Optional[str] = None,
        model_type: str = "classification",
        project_name: Optional[str] = None
    ):
        """
        Initialize DriftGuard with model and reference data.
        
        Args:
            model: The model to monitor
            reference_data: Reference data for drift detection
            config_path: Path to configuration file (optional)
            model_type: Type of model ('classification' or 'regression')
            project_name: Name for this monitoring instance
        """
        self.model = model
        self.model_type = model_type
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        if project_name:
            self.config_manager.update_config({"project_name": project_name})
        
        # Initialize components
        self._initialize_components(reference_data)
        
        logger.info(
            f"DriftGuard initialized for project "
            f"'{self.config_manager.config.project_name}'"
        )
    
    def _initialize_components(self, reference_data: pd.DataFrame) -> None:
        """Initialize all monitoring components"""
        # Create state manager
        self.state_manager = StateManager(self.config_manager.config.storage)
        
        # Create data validator
        self.schema = DataSchema.from_dataframe(reference_data)
        self.validator = DataValidator(self.schema)
        
        # Create drift detector
        self.drift_detector = create_drift_detector(
            self.config_manager.config.drift.method,
            self.config_manager.config.drift
        )
        self.drift_detector.initialize(reference_data)
        
        # Create model monitor
        self.model_monitor = create_model_monitor(
            self.model_type,
            self.config_manager.config,
            self.state_manager
        )
        
        # Create alert manager
        self.alert_manager = AlertManager(self.config_manager.config.alerts)
        
        # Initialize thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
    
    async def monitor_batch(
        self,
        features: pd.DataFrame,
        actual_labels: Optional[Union[pd.Series, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raise_on_drift: bool = False
    ) -> Dict[str, Any]:
        """
        Monitor a batch of data for drift and model performance.
        
        Args:
            features: Input features to monitor
            actual_labels: True labels if available
            metadata: Additional metadata to store
            raise_on_drift: Whether to raise exception on drift detection
            
        Returns:
            Dictionary containing monitoring results
        """
        # Validate input data
        is_valid, messages = self.validator.validate(features)
        if not is_valid:
            error_msg = "Data validation failed:\n" + "\n".join(messages)
            logger.error(error_msg)
            if raise_on_drift:
                raise ValueError(error_msg)
            return {
                "status": "error",
                "messages": messages
            }
        
        results = {
            "status": "success",
            "timestamp": datetime.now(),
            "sample_size": len(features),
            "drift_detected": False,
            "performance_metrics": None,
            "drift_reports": [],
            "warnings": []
        }
        
        try:
            # Detect drift
            drift_reports = await self._detect_drift_async(features)
            results["drift_reports"] = [
                report.__dict__ for report in drift_reports
            ]
            
            # Check if drift detected
            drift_detected = any(
                report.drift_score > self.config_manager.config.drift.threshold
                for report in drift_reports
            )
            results["drift_detected"] = drift_detected
            
            if drift_detected:
                await self._handle_drift_detection(drift_reports)
                if raise_on_drift:
                    raise ValueError("Drift detected in input data")
            
            # Monitor performance if labels available
            if actual_labels is not None:
                performance_metrics = await self._monitor_performance_async(
                    features, actual_labels, metadata
                )
                results["performance_metrics"] = performance_metrics
        
        except Exception as e:
            error_msg = f"Monitoring failed: {str(e)}"
            logger.error(error_msg)
            results["status"] = "error"
            results["messages"] = [error_msg]
        
        # Update state
        self._update_monitoring_state(results)
        
        return results
    
    async def _detect_drift_async(
        self,
        features: pd.DataFrame
    ) -> List[DriftReport]:
        """Detect drift asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.drift_detector.detect_drift,
            features
        )
    
    async def _monitor_performance_async(
        self,
        features: pd.DataFrame,
        actual_labels: Union[pd.Series, np.ndarray],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Monitor model performance asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Get predictions
        predictions = await loop.run_in_executor(
            self.thread_pool,
            self.model.predict,
            features
        )
        
        # Track performance
        return await loop.run_in_executor(
            self.thread_pool,
            self.model_monitor.track_performance,
            features,
            predictions,
            actual_labels,
            metadata
        )
    
    async def _handle_drift_detection(
        self,
        drift_reports: List[DriftReport]
    ) -> None:
        """Handle drift detection results"""
        # Prepare drift summary
        drifted_features = [
            report.feature_name for report in drift_reports
            if report.drift_score > self.config_manager.config.drift.threshold
        ]
        
        drift_scores = {
            report.feature_name: report.drift_score
            for report in drift_reports
        }
        
        # Create alert message
        message = (
            f"Drift detected in {len(drifted_features)} features:\n"
            + "\n".join(
                f"- {feature}: {drift_scores[feature]:.3f}"
                for feature in drifted_features
            )
        )
        
        # Send alert
        await self.alert_manager.send_alert(
            alert_type="drift_detected",
            message=message,
            severity=AlertLevel.WARNING,
            metadata={
                "drifted_features": drifted_features,
                "drift_scores": drift_scores
            }
        )
    
    def _update_monitoring_state(self, results: Dict[str, Any]) -> None:
        """Update monitoring state"""
        try:
            self.state_manager.save_state({
                "last_update": results["timestamp"].isoformat(),
                "monitoring": {
                    "total_samples_processed": (
                        self.state_manager.load_state()
                        .get("monitoring", {})
                        .get("total_samples_processed", 0)
                        + results["sample_size"]
                    ),
                    "last_drift_detected": (
                        results["timestamp"].isoformat()
                        if results["drift_detected"]
                        else None
                    )
                }
            })
        except Exception as e:
            logger.error(f"Failed to update state: {str(e)}")
    
    def get_monitoring_summary(
        self,
        include_drift_history: bool = True,
        include_performance_history: bool = True
    ) -> Dict[str, Any]:
        """Get summary of monitoring results"""
        summary = {
            "project": self.config_manager.config.project_name,
            "model_type": self.model_type,
            "status": self.state_manager.get_system_status()
        }
        
        if include_drift_history:
            summary["drift_history"] = self.state_manager.get_drift_history()
            
        if include_performance_history:
            summary["performance"] = self.model_monitor.get_performance_summary()
            
        return summary
    
    def update_reference_data(
        self,
        new_reference: pd.DataFrame,
        validate: bool = True
    ) -> None:
        """Update reference data for drift detection"""
        if validate:
            is_valid, messages = self.validator.validate(new_reference)
            if not is_valid:
                raise ValueError(
                    "Invalid reference data:\n" + "\n".join(messages)
                )
        
        self.drift_detector.update_reference(new_reference)
        logger.info("Updated reference data for drift detection")
    
    def update_config(
        self,
        updates: Dict[str, Any],
        reinitialize: bool = True
    ) -> None:
        """Update configuration"""
        self.config_manager.update_config(updates)
        
        if reinitialize:
            self._initialize_components(self.drift_detector.reference_data)
            
        logger.info("Updated configuration and reinitialized components")
    
    async def start_monitoring_service(
        self,
        data_generator: Any,
        interval_seconds: int = 60,
        max_iterations: Optional[int] = None
    ) -> None:
        """
        Start continuous monitoring service.
        
        Args:
            data_generator: Generator/iterator providing monitoring data
            interval_seconds: Monitoring interval in seconds
            max_iterations: Maximum number of monitoring iterations (optional)
        """
        iteration = 0
        
        while True:
            if max_iterations and iteration >= max_iterations:
                break
                
            try:
                # Get next batch of data
                batch = next(data_generator)
                features = batch.get("features")
                labels = batch.get("labels")
                metadata = batch.get("metadata")
                
                if features is not None:
                    await self.monitor_batch(features, labels, metadata)
                
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Monitoring iteration failed: {str(e)}")
                
            iteration += 1
            await asyncio.sleep(interval_seconds)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.thread_pool.shutdown(wait=True)
        logger.info("DriftGuard monitoring stopped")
