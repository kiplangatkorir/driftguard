"""
Core interfaces and data models for DriftGuard.
"""
from typing import Dict, List, Optional, Protocol
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

class DriftReport(BaseModel):
    """Report containing drift detection results"""
    feature_name: str
    drift_score: float
    p_value: Optional[float] = None
    threshold: float
    method: str
    has_drift: bool

class IDriftDetector(Protocol):
    """Interface for drift detection"""
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        ...
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        ...
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        ...

class IStateManager(Protocol):
    """Interface for state management"""
    
    def save_state(self, state: Dict) -> None:
        """Save current state"""
        ...
    
    def load_state(self) -> Dict:
        """Load saved state"""
        ...
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update metrics history"""
        ...
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history"""
        ...
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        ...

class IAlertManager(Protocol):
    """Interface for alert management"""
    
    def create_alert(
        self,
        message: str,
        alert_type: str,
        severity: str = "info",
        metadata: Optional[Dict] = None
    ) -> None:
        """Create new alert"""
        ...
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get filtered alerts"""
        ...
    
    def clear_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """Clear alerts matching criteria"""
        ...

class IModelMonitor(Protocol):
    """Interface for model monitoring"""
    
    def initialize(self, model_type: str) -> None:
        """Initialize monitor for specific model type"""
        ...
    
    def track_performance(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Track model performance"""
        ...
    
    def check_degradation(
        self,
        metric: str,
        window: Optional[int] = None
    ) -> bool:
        """Check for performance degradation"""
        ...
    
    def get_performance_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get performance metrics history"""
        ...
