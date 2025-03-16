"""
Configuration management for DriftGuard using Pydantic v2.
"""
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DriftMethod(str, Enum):
    """Supported drift detection methods"""
    KS_TEST = "ks_test"
    JSD = "jensen_shannon"
    PSI = "psi"

class StorageConfig(BaseModel):
    """Storage configuration"""
    type: str = Field(default="local", description="Storage type (local, s3)")
    path: str = Field(default="./data", description="Storage path")
    retention_days: int = Field(default=30, description="Data retention period in days")

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    method: DriftMethod = Field(default=DriftMethod.KS_TEST, description="Drift detection method")
    threshold: float = Field(default=0.05, description="Drift detection threshold")
    window_size: int = Field(default=1000, description="Window size for drift detection")
    reference_update_frequency: Optional[int] = Field(default=None, description="Auto-update reference data frequency")

class MonitorConfig(BaseModel):
    """Model monitoring configuration"""
    metrics: List[str] = Field(
        default=["accuracy", "f1", "roc_auc"],
        description="Performance metrics to track"
    )
    window_size: int = Field(default=1000, description="Window size for metrics")
    performance_threshold: float = Field(default=0.1, description="Performance degradation threshold")

class AlertConfig(BaseModel):
    """Alert configuration"""
    enabled: bool = Field(default=True, description="Enable/disable alerts")
    channels: List[str] = Field(default=["email"], description="Alert channels")
    email_recipients: Optional[List[str]] = Field(default=None, description="Email recipients")
    min_interval_minutes: int = Field(default=60, description="Minimum time between alerts")

class DriftGuardConfig(BaseModel):
    """Main configuration for DriftGuard"""
    version: str = Field(default="0.1.3", description="DriftGuard version")
    project_name: str = Field(default="default", description="Project name")
    storage: StorageConfig = Field(default_factory=StorageConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DriftGuardConfig':
        """Load configuration from YAML file"""
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.model_validate(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {str(e)}")
            raise
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        try:
            config_dict = self.model_dump()
            with open(path, 'w') as f:
                yaml.dump(config_dict, f)
            logger.info(f"Saved config to {path}")
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {str(e)}")
            raise
    
    def update(self, updates: Dict) -> None:
        """Update configuration with new values"""
        try:
            # Create new config with updates
            updated = self.model_copy(update=updates)
            
            # Update self with new values
            for key, value in updated.model_dump().items():
                setattr(self, key, value)
            
            logger.info("Updated configuration")
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            raise
    
    @field_validator('version')
    def validate_version(cls, v):
        """Validate version format"""
        from packaging import version
        try:
            version.parse(v)
            return v
        except version.InvalidVersion:
            raise ValueError(f"Invalid version format: {v}")
    
    @field_validator('drift')
    def validate_drift_config(cls, v):
        """Validate drift configuration"""
        if v.threshold <= 0 or v.threshold >= 1:
            raise ValueError("Drift threshold must be between 0 and 1")
        if v.window_size < 100:
            raise ValueError("Window size must be at least 100")
        return v
    
    @field_validator('monitor')
    def validate_monitor_config(cls, v):
        """Validate monitor configuration"""
        valid_metrics = {
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "mse", "rmse", "mae"
        }
        invalid_metrics = set(v.metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
        return v
    
    @field_validator('alerts')
    def validate_alert_config(cls, v):
        """Validate alert configuration"""
        valid_channels = {"email", "slack", "webhook"}
        invalid_channels = set(v.channels) - valid_channels
        if invalid_channels:
            raise ValueError(f"Invalid alert channels: {invalid_channels}")
        
        if "email" in v.channels and not v.email_recipients:
            raise ValueError("Email recipients required when email alerts enabled")
        return v
