"""
Core interfaces for DriftGuard components.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd

class DriftReport(BaseModel):
    """Report containing drift detection results"""
    feature_name: str
    drift_score: float
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    threshold: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class AlertConfig(BaseModel):
    """Configuration for alert channels"""
    email_enabled: bool = False
    email_recipients: Optional[List[str]] = None
    email_smtp_host: Optional[str] = None
    email_smtp_port: Optional[int] = None
    email_smtp_user: Optional[str] = None
    email_smtp_password: Optional[str] = None
    slack_enabled: bool = False
    slack_webhook: Optional[str] = None
    
    @field_validator('slack_webhook')
    def validate_slack_webhook(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith('https://hooks.slack.com/'):
            raise ValueError('Invalid Slack webhook URL')
        return v
    
    @field_validator('email_recipients')
    def validate_email_recipients(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for email in v:
                if '@' not in email or '.' not in email:
                    raise ValueError(f'Invalid email address: {email}')
        return v

class MetricConfig(BaseModel):
    """Configuration for performance metrics"""
    name: str
    threshold: float = 0.1
    window_size: Optional[int] = None

class DriftConfig(BaseModel):
    """Configuration for drift detection"""
    method: str = 'ks_test'
    threshold: float = 0.05
    window_size: Optional[int] = None
    features: Optional[List[str]] = None

class StorageConfig(BaseModel):
    """Configuration for state storage"""
    type: str = 'local'
    path: Optional[str] = None
    retention_days: int = 30

class MonitoringConfig(BaseModel):
    """Main configuration for DriftGuard"""
    project_name: str
    model_type: str = 'classification'
    metrics: List[MetricConfig]
    drift: DriftConfig
    alerts: AlertConfig
    storage: StorageConfig

class IDriftDetector(ABC):
    """Interface for drift detection"""
    
    @abstractmethod
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize with reference data"""
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
    """Interface for model monitoring"""
    
    @abstractmethod
    def track_performance(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actual: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Track model performance"""
        pass

class IStateManager(ABC):
    """Interface for state management"""
    
    @abstractmethod
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save state"""
        pass
    
    @abstractmethod
    def load_state(self) -> Dict[str, Any]:
        """Load state"""
        pass
    
    @abstractmethod
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """Update metrics history"""
        pass
    
    @abstractmethod
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history"""
        pass
    
    @abstractmethod
    def add_warning(self, message: str) -> None:
        """Add warning message"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        pass

class IAlertManager(ABC):
    """Interface for alert management"""
    
    @abstractmethod
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send alert"""
        pass
