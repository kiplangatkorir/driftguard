"""
Core interfaces for DriftGuard components.
Defines the contract that all implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DriftReport:
    """Standardized drift detection report"""
    feature_name: str
    drift_score: float
    p_value: float
    timestamp: datetime
    sample_size: int
    drift_type: str
    additional_metrics: Dict[str, Any]

@dataclass
class MonitoringConfig:
    """Configuration for monitoring components"""
    drift_threshold: float = 0.05
    performance_threshold: float = 0.1
    monitoring_window: int = 1000
    batch_size: Optional[int] = None
    feature_importance_threshold: float = 0.1
    alert_cooldown: int = 3600  # seconds
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "f1", "roc_auc"]

class IDriftDetector(ABC):
    """Interface for drift detection implementations"""
    
    @abstractmethod
    def initialize(self, reference_data: pd.DataFrame, config: MonitoringConfig) -> None:
        """Initialize the detector with reference data"""
        pass
    
    @abstractmethod
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        pass
    
    @abstractmethod
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        pass

class IModelMonitor(ABC):
    """Interface for model monitoring implementations"""
    
    @abstractmethod
    def track_performance(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray
    ) -> Dict[str, float]:
        """Track model performance metrics"""
        pass
    
    @abstractmethod
    def get_performance_history(self) -> pd.DataFrame:
        """Get historical performance data"""
        pass

class IAlertManager(ABC):
    """Interface for alert management implementations"""
    
    @abstractmethod
    def send_alert(self, alert_type: str, message: str, severity: str) -> bool:
        """Send an alert"""
        pass
    
    @abstractmethod
    def should_alert(self, alert_type: str) -> bool:
        """Check if an alert should be sent based on cooldown"""
        pass

class IDataValidator(ABC):
    """Interface for data validation implementations"""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the expected data schema"""
        pass

class IStateManager(ABC):
    """Interface for state management implementations"""
    
    @abstractmethod
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        pass
    
    @abstractmethod
    def load_state(self) -> Dict[str, Any]:
        """Load saved state"""
        pass
    
    @abstractmethod
    def clear_state(self) -> None:
        """Clear saved state"""
        pass
