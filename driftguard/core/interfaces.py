"""
Core interfaces and data models for DriftGuard.
"""
from typing import Any, Dict, List, Optional, Protocol, Union
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

class DriftReport(BaseModel):
    """Data model for drift detection report"""
    model_config = ConfigDict(extra='forbid')
    
    feature_name: str = Field(description="Name of the feature")
    drift_score: float = Field(description="Drift score")
    threshold: float = Field(description="Drift threshold")
    p_value: Optional[float] = Field(default=None, description="Statistical test p-value")
    test_statistic: Optional[float] = Field(default=None, description="Test statistic")
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    
    @property
    def has_drift(self) -> bool:
        """Check if drift is detected"""
        if self.p_value is not None:
            return self.p_value < self.threshold
        return self.drift_score > self.threshold

class MetricReport(BaseModel):
    """Data model for metric report"""
    model_config = ConfigDict(extra='forbid')
    
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    threshold: float = Field(description="Alert threshold")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")
    
    @property
    def exceeds_threshold(self) -> bool:
        """Check if metric exceeds threshold"""
        return abs(self.value) > self.threshold

class IDriftDetector(Protocol):
    """Interface for drift detection"""
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
        ...
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        ...
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftReport]:
        """Detect drift in new data"""
        ...

class IStateManager(Protocol):
    """Interface for state management"""
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        ...
    
    def load_state(self) -> Dict[str, Any]:
        """Load current state"""
        ...
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
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
    
    def add_warning(self, message: str) -> None:
        """Add warning message"""
        ...
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status summary"""
        ...

class IAlertManager(Protocol):
    """Interface for alert management"""
    
    def send_drift_alert(
        self,
        drift_reports: List[DriftReport],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send drift detection alert"""
        ...
    
    def send_metric_alert(
        self,
        metric_reports: List[MetricReport],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send metric alert"""
        ...
    
    def send_system_alert(
        self,
        message: str,
        level: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send system alert"""
        ...

class IModelMonitor(Protocol):
    """Interface for model monitoring"""
    
    def initialize(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        reference_labels: Optional[pd.Series] = None
    ) -> None:
        """Initialize monitor with reference data"""
        ...
    
    def update_reference(
        self,
        new_reference: pd.DataFrame,
        new_labels: Optional[pd.Series] = None
    ) -> None:
        """Update reference data"""
        ...
    
    def track_performance(
        self,
        new_data: pd.DataFrame,
        actual_labels: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None
    ) -> List[MetricReport]:
        """Track model performance"""
        ...
    
    def check_performance(
        self,
        window_size: Optional[int] = None
    ) -> List[MetricReport]:
        """Check model performance over time window"""
        ...

class IDataValidator(Protocol):
    """Interface for data validation"""
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        ...
    
    def validate_schema(self, data: pd.DataFrame) -> List[str]:
        """Validate data schema"""
        ...
    
    def validate_ranges(self, data: pd.DataFrame) -> List[str]:
        """Validate data ranges"""
        ...
    
    def validate_missing(self, data: pd.DataFrame) -> List[str]:
        """Validate missing values"""
        ...
    
    def validate_types(self, data: pd.DataFrame) -> List[str]:
        """Validate data types"""
        ...
