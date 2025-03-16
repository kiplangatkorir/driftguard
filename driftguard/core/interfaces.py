"""
Core interfaces and data models for DriftGuard components.
"""
from typing import Any, Dict, List, Optional, Protocol, Union
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

class DriftReport(BaseModel):
    """Report for drift detection results"""
    feature_name: str = Field(description="Name of the feature")
    drift_score: float = Field(description="Drift detection score")
    p_value: Optional[float] = Field(default=None, description="Statistical test p-value")
    threshold: float = Field(description="Drift detection threshold")
    method: str = Field(description="Drift detection method used")
    timestamp: datetime = Field(default_factory=datetime.now)
    has_drift: bool = Field(description="Whether drift was detected")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class MetricReport(BaseModel):
    """Report for model performance metrics"""
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Current metric value")
    threshold: float = Field(description="Performance threshold")
    exceeds_threshold: bool = Field(description="Whether threshold is exceeded")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

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
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        ...
    
    def load_state(self) -> Dict[str, Any]:
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        ...

class IAlertManager(Protocol):
    """Interface for alert management"""
    async def send_drift_alert(
        self,
        drift_reports: List[DriftReport],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send drift alert"""
        ...
    
    async def send_metric_alert(
        self,
        metric_reports: List[MetricReport],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send metric alert"""
        ...

class IModelMonitor(Protocol):
    """Interface for model monitoring"""
    def initialize(
        self,
        model: Any,
        reference_data: pd.DataFrame
    ) -> None:
        """Initialize monitor"""
        ...
    
    def track_performance(
        self,
        data: pd.DataFrame,
        actual_labels: Union[pd.Series, np.ndarray]
    ) -> List[MetricReport]:
        """Track model performance"""
        ...
    
    def update_reference(
        self,
        new_reference: pd.DataFrame,
        new_labels: Optional[pd.Series] = None
    ) -> None:
        """Update reference data"""
        ...
