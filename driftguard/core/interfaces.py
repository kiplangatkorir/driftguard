"""
Core interfaces and data models for DriftGuard components.
"""
from typing import Any, Dict, List, Optional, Protocol
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field

class DriftReport(BaseModel):
    """Drift detection report"""
    feature_name: str = Field(description="Name of the feature")
    drift_score: float = Field(description="Drift score")
    p_value: Optional[float] = Field(
        default=None,
        description="P-value for statistical tests"
    )
    threshold: float = Field(description="Drift threshold")
    method: str = Field(description="Drift detection method used")
    has_drift: bool = Field(description="Whether drift was detected")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Time of detection"
    )

class MetricReport(BaseModel):
    """Model performance metric report"""
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    threshold: float = Field(description="Performance threshold")
    exceeds_threshold: bool = Field(
        description="Whether metric exceeds threshold"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Time of measurement"
    )

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
        actual_labels: pd.Series
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

class IDataValidator(Protocol):
    """Interface for data validation"""
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        ...
    
    def validate(self, data: pd.DataFrame) -> Any:
        """Validate input data"""
        ...
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        ...
