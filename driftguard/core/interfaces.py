"""
Core interfaces and data models for DriftGuard.
"""
from typing import Dict, List, Optional, Protocol, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        timestamp: Optional[datetime] = None
    ):
        self.method = method
        self.score = score
        self.threshold = threshold
        self.features = features
        self.timestamp = timestamp or datetime.now()
        self.has_drift = score > threshold

@dataclass
class MetricReport:
    """Report for model performance metrics."""
    metric: str
    value: float
    threshold: float
    is_degraded: bool
    history: List[float]

@dataclass
class AlertReport:
    """Report for alert status."""
    alert_type: str
    message: str
    timestamp: str
    severity: str
    is_sent: bool

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

class IDriftDetectorABC(ABC):
    """Interface for drift detection."""
    
    @abstractmethod
    def detect_drift(
        self,
        new_data: pd.DataFrame,
        methods: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Detect drift in new data."""
        pass

class IModelMonitor(Protocol):
    """Interface for model monitoring"""
    def initialize(
        self,
        reference_predictions: pd.Series,
        reference_labels: pd.Series
    ) -> None:
        """Initialize monitor with reference data"""
        ...
    
    def track(
        self,
        predictions: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Track model performance"""
        ...

class IModelMonitorABC(ABC):
    """Interface for model monitoring."""
    
    @abstractmethod
    def track_performance(
        self,
        data: pd.DataFrame,
        labels: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Track model performance."""
        pass
    
    @abstractmethod
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get performance history."""
        pass

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

class IAlertManagerABC(ABC):
    """Interface for alert management."""
    
    @abstractmethod
    def send_alert(
        self,
        message: str,
        alert_type: str = "drift",
        severity: str = "warning"
    ) -> bool:
        """Send an alert."""
        pass
    
    @abstractmethod
    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        pass
    
    @abstractmethod
    def get_alert_history(self) -> List[AlertReport]:
        """Get alert history."""
        pass

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

class IStateManagerABC(ABC):
    """Interface for state management."""
    
    @abstractmethod
    def save_version(
        self,
        model: Any,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a model version."""
        pass
    
    @abstractmethod
    def load_version(self, version: str) -> Any:
        """Load a model version."""
        pass
    
    @abstractmethod
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        pass
