"""
Core interfaces and data models for DriftGuard.
"""
from typing import Dict, List, Optional, Protocol
from enum import Enum
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

class DriftMethod(str, Enum):
    """Drift detection methods"""
    KS_TEST = "ks_test"
    JSD = "jsd"
    PSI = "psi"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DriftReport(BaseModel):
    """Drift detection report"""
    feature: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    threshold: float
    has_drift: bool
    timestamp: datetime

class ValidationResult(BaseModel):
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class Alert(BaseModel):
    """Alert model"""
    id: str
    message: str
    alert_type: str
    severity: AlertSeverity
    timestamp: datetime
    metadata: Dict

class IDriftDetector(Protocol):
    """Interface for drift detectors"""
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize detector with reference data"""
        ...
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        ...

class IModelMonitor(Protocol):
    """Interface for model monitors"""
    def initialize(self, model_type: str) -> None:
        """Initialize monitor"""
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

class IDataValidator(Protocol):
    """Interface for data validators"""
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        ...
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data"""
        ...

class IStateManager(Protocol):
    """Interface for state managers"""
    def save_state(self, state: Dict) -> None:
        """Save state"""
        ...
    
    def load_state(self) -> Dict:
        """Load state"""
        ...
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics"""
        ...
    
    def get_metrics_history(self) -> pd.DataFrame:
        """Get metrics history"""
        ...

class IAlertManager(Protocol):
    """Interface for alert managers"""
    def create_alert(
        self,
        message: str,
        alert_type: str,
        severity: AlertSeverity,
        metadata: Optional[Dict] = None
    ) -> Alert:
        """Create alert"""
        ...
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get alerts"""
        ...
    
    def clear_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> int:
        """Clear alerts"""
        ...
