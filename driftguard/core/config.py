"""
Configuration management for DriftGuard using Pydantic v2.
Handles loading and validation of configuration from YAML and environment variables.
"""
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml
import os
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
import logging

logger = logging.getLogger(__name__)

class AlertConfig(BaseModel):
    """Alert configuration settings"""
    model_config = ConfigDict(extra='forbid')
    
    enabled: bool = Field(default=True, description="Enable/disable alerts")
    email_recipients: List[str] = Field(default_factory=list, description="List of email recipients")
    slack_webhook: Optional[str] = Field(default=None, description="Slack webhook URL")
    alert_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Alert threshold for drift detection"
    )
    
    @field_validator('email_recipients')
    @classmethod
    def validate_emails(cls, emails: List[str]) -> List[str]:
        """Validate email addresses"""
        for email in emails:
            if '@' not in email or '.' not in email:
                raise ValueError(f"Invalid email address: {email}")
        return emails

class StorageConfig(BaseModel):
    """Storage configuration settings"""
    model_config = ConfigDict(extra='forbid')
    
    path: str = Field(
        default="./storage",
        description="Path to storage directory"
    )
    retention_days: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of days to retain data"
    )
    compression: bool = Field(
        default=True,
        description="Enable data compression"
    )
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Validate storage path"""
        path_obj = Path(path)
        if not path_obj.parent.exists():
            raise ValueError(f"Parent directory does not exist: {path_obj.parent}")
        return str(path_obj.absolute())

class MonitorConfig(BaseModel):
    """Model monitoring configuration"""
    model_config = ConfigDict(extra='forbid')
    
    metrics: List[str] = Field(
        default=["accuracy", "f1_score"],
        description="List of metrics to monitor"
    )
    window_size: int = Field(
        default=1000,
        ge=10,
        description="Window size for monitoring"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Batch size for processing"
    )
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, metrics: List[str]) -> List[str]:
        """Validate metric names"""
        valid_metrics = {
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "mse", "rmse", "mae"
        }
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric: {metric}. "
                    f"Valid metrics: {valid_metrics}"
                )
        return metrics

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    model_config = ConfigDict(extra='forbid')
    
    method: str = Field(
        default="ks_test",
        description="Drift detection method"
    )
    threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Drift detection threshold"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Features to monitor for drift"
    )
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, method: str) -> str:
        """Validate drift detection method"""
        valid_methods = {"ks_test", "jsd", "psi"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid drift detection method: {method}. "
                f"Valid methods: {valid_methods}"
            )
        return method

class DriftGuardConfig(BaseModel):
    """Main configuration for DriftGuard"""
    model_config = ConfigDict(extra='forbid')
    
    version: str = Field(
        default="0.1.3",
        description="DriftGuard version"
    )
    project_name: str = Field(
        description="Name of the monitoring project"
    )
    model_id: str = Field(
        description="Unique identifier for the model"
    )
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alert configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration"
    )
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig,
        description="Monitoring configuration"
    )
    drift: DriftConfig = Field(
        default_factory=DriftConfig,
        description="Drift detection configuration"
    )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DriftGuardConfig":
        """Load configuration from YAML file"""
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.model_validate(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {str(e)}")
            raise
    
    @classmethod
    def from_env(cls) -> "DriftGuardConfig":
        """Load configuration from environment variables"""
        env_prefix = "DRIFTGUARD_"
        config_dict = {}
        
        # Map environment variables to config structure
        env_mapping = {
            "PROJECT_NAME": "project_name",
            "MODEL_ID": "model_id",
            "ALERT_ENABLED": "alerts.enabled",
            "ALERT_EMAILS": "alerts.email_recipients",
            "ALERT_SLACK": "alerts.slack_webhook",
            "STORAGE_PATH": "storage.path",
            "RETENTION_DAYS": "storage.retention_days",
            "DRIFT_METHOD": "drift.method",
            "DRIFT_THRESHOLD": "drift.threshold"
        }
        
        for env_key, config_key in env_mapping.items():
            env_value = os.getenv(env_prefix + env_key)
            if env_value is not None:
                # Handle special cases
                if env_key == "ALERT_EMAILS":
                    env_value = env_value.split(",")
                elif env_key in ["ALERT_ENABLED"]:
                    env_value = env_value.lower() == "true"
                elif env_key in ["RETENTION_DAYS"]:
                    env_value = int(env_value)
                elif env_key in ["DRIFT_THRESHOLD"]:
                    env_value = float(env_value)
                
                # Build nested dictionary
                parts = config_key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = env_value
        
        return cls.model_validate(config_dict)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        config_dict = self.model_dump()
        
        def update_nested(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
            for key, value in updates.items():
                if isinstance(value, dict) and key in base:
                    update_nested(base[key], value)
                else:
                    base[key] = value
        
        update_nested(config_dict, updates)
        new_config = self.model_validate(config_dict)
        
        # Update all fields
        for field in self.model_fields:
            setattr(self, field, getattr(new_config, field))
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(
                    self.model_dump(),
                    f,
                    default_flow_style=False
                )
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {str(e)}")
            raise
