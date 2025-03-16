"""
Configuration module for DriftGuard.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr, validator
import logging

class DriftConfig(BaseModel):
    """Configuration for drift detection"""
    threshold: float = Field(
        default=0.05,
        description="Threshold for drift detection"
    )
    min_samples: int = Field(
        default=100,
        description="Minimum number of samples required"
    )
    methods: List[str] = Field(
        default=["ks_test", "jsd", "psi"],
        description="List of drift detection methods to use"
    )
    window_size: Optional[int] = Field(
        default=None,
        description="Window size for drift detection"
    )
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0 < v < 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v
    
    @validator('min_samples')
    def validate_min_samples(cls, v):
        if v < 10:
            raise ValueError("Minimum samples must be at least 10")
        return v

class MonitorConfig(BaseModel):
    """Configuration for model monitoring"""
    metrics: List[str] = Field(
        default=[
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "mse", "rmse", "mae", "r2"
        ],
        description="List of metrics to track"
    )
    degradation_threshold: float = Field(
        default=0.1,
        description="Threshold for performance degradation"
    )
    window_size: Optional[int] = Field(
        default=None,
        description="Window size for performance tracking"
    )
    
    @validator('degradation_threshold')
    def validate_degradation_threshold(cls, v):
        if not 0 < v < 1:
            raise ValueError("Degradation threshold must be between 0 and 1")
        return v

class AlertConfig(BaseModel):
    """Configuration for alert management"""
    email_enabled: bool = Field(
        default=False,
        description="Whether to enable email alerts"
    )
    email_sender: Optional[EmailStr] = Field(
        default=None,
        description="Email address to send alerts from"
    )
    email_recipients: List[EmailStr] = Field(
        default_factory=list,
        description="List of email recipients"
    )
    email_severity_levels: List[str] = Field(
        default=["warning", "critical"],
        description="Severity levels that trigger email alerts"
    )
    smtp_host: str = Field(
        default="smtp.gmail.com",
        description="SMTP server host"
    )
    smtp_port: int = Field(
        default=587,
        description="SMTP server port"
    )
    smtp_username: Optional[str] = Field(
        default=None,
        description="SMTP username"
    )
    smtp_password: Optional[str] = Field(
        default=None,
        description="SMTP password"
    )
    smtp_use_tls: bool = Field(
        default=True,
        description="Whether to use TLS for SMTP"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Path to log file"
    )
    log_level: int = Field(
        default=logging.INFO,
        description="Logging level"
    )
    
    @validator('smtp_port')
    def validate_smtp_port(cls, v):
        if not 0 <= v <= 65535:
            raise ValueError("Invalid SMTP port number")
        return v

class StorageConfig(BaseModel):
    """Configuration for state storage"""
    path: str = Field(
        default="./storage",
        description="Path to storage directory"
    )
    retention_days: int = Field(
        default=30,
        description="Number of days to retain metrics"
    )
    
    @validator('retention_days')
    def validate_retention_days(cls, v):
        if v < 1:
            raise ValueError("Retention days must be at least 1")
        return v

class DriftGuardConfig(BaseModel):
    """Main configuration for DriftGuard"""
    drift: DriftConfig = Field(
        default_factory=DriftConfig,
        description="Drift detection configuration"
    )
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig,
        description="Model monitoring configuration"
    )
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alert management configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration"
    )
    
    @classmethod
    def from_yaml(cls, path: str) -> "DriftGuardConfig":
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""
        import yaml
        config_dict = self.dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
