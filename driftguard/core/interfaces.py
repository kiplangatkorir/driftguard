"""
Core interfaces and data models for DriftGuard.
"""
from typing import Dict, List, Optional, Protocol, Any
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod

class ValidationResult:
    """Result of data validation"""
    def __init__(
        self,
        is_valid: bool,
        errors: List[str],
        warnings: List[str]
    ):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings

class DriftReport:
    """Report of drift detection results"""
    def __init__(
        self,
        method: str,
        score: float,
        threshold: float,
        features: List[str],
        importance_change: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.method = method
        self.score = score
        self.threshold = threshold
        self.features = features
        self.importance_change = importance_change
        self.timestamp = timestamp or datetime.now()
        self.has_drift = score > threshold

class IDataValidator(Protocol):
    """Interface for data validation"""
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        ...
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data quality"""
        ...

class IDriftDetector(Protocol):
    """Interface for drift detection"""
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize detector with reference data"""
        ...
    
    def detect(self, data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        ...

class IDriftDetectorParallel(IDriftDetector, ABC):
    """Interface for parallel drift detection"""
    @abstractmethod
    def detect(self, data: pd.DataFrame, parallel: bool = False) -> List[DriftReport]:
        """Detect drift in new data
        
        Args:
            data: New data to analyze
            parallel: Whether to use parallel processing
        """
        pass

class IModelMonitor(Protocol):
    """Interface for model monitoring"""
    def initialize(
        self,
        reference_predictions: pd.Series,
        reference_labels: pd.Series,
        model: Optional[Any] = None
    ) -> None:
        """Initialize monitor with reference data"""
        ...
        
    def attach_model(self, model: Any) -> None:
        """Attach a model for potential retraining"""
        ...
    
    def attach_alert_manager(self, alert_manager: Any) -> None:
        """Attach an AlertManager for performance alerts"""
        ...
        
    def should_retrain(self, current_performance: dict) -> bool:
        """Determine if retraining is needed"""
        ...
        
    def retrain_model(self, X_new: pd.DataFrame, y_new: pd.Series) -> Any:
        """Retrain attached model with new data"""
        ...

class IStateManager(Protocol):
    """Interface for state management"""
    def save_state(self, state: Dict) -> None:
        """Save current state"""
        ...
    
    def load_state(self) -> Dict:
        """Load saved state"""
        ...
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics history"""
        ...
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history"""
        ...

class IAlertManager(Protocol):
    """Interface for alert management"""
    def add_alert(
        self,
        message: str,
        severity: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add new alert"""
        ...
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List:
        """Get filtered alerts"""
        ...
    
    def clear_alerts(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None
    ) -> None:
        """Clear alerts matching filters"""
        ...
