"""
Configuration management for DriftGuard.
Handles all configuration settings with validation and environment variable support.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, EmailStr, validator, Field
import os
from enum import Enum
import yaml
import logging
from pathlib import Path

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class DriftMethod(str, Enum):
    """Supported drift detection methods"""
    KS_TEST = "ks_test"
    CHI_SQUARE = "chi_square"
    MMD = "maximum_mean_discrepancy"
    ISOLATION_FOREST = "isolation_forest"
    ADWIN = "adaptive_windowing"

class MetricConfig(BaseModel):
    """Configuration for individual metrics"""
    name: str
    threshold: float = Field(gt=0, lt=1)
    comparison_window: int = Field(gt=0)
    
    class Config:
        extra = "forbid"

class AlertConfig(BaseModel):
    """Alert system configuration"""
    email_recipients: List[EmailStr] = []
    slack_webhook: Optional[str] = None
    cooldown_period: int = Field(default=3600, gt=0)  # seconds
    min_severity: AlertLevel = AlertLevel.WARNING
    
    @validator('slack_webhook')
    def validate_webhook(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Slack webhook must be a valid URL')
        return v

class StorageConfig(BaseModel):
    """Storage configuration for monitoring data"""
    storage_type: str = "local"  # local, s3, gcs, azure
    path: str = "./monitoring_data"
    retention_days: int = Field(default=30, gt=0)
    compression: bool = True
    
    class Config:
        extra = "forbid"

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    method: DriftMethod = DriftMethod.KS_TEST
    threshold: float = Field(default=0.05, gt=0, lt=1)
    window_size: int = Field(default=1000, gt=0)
    feature_importance_threshold: float = Field(default=0.1, gt=0, lt=1)
    batch_size: Optional[int] = Field(default=None, gt=0)
    
    class Config:
        extra = "forbid"

class MonitoringConfig(BaseModel):
    """Main configuration for DriftGuard"""
    project_name: str
    model_name: str
    version: str
    drift: DriftConfig
    metrics: List[MetricConfig]
    alerts: AlertConfig
    storage: StorageConfig
    log_level: str = "INFO"
    enable_async: bool = False
    
    class Config:
        extra = "forbid"

class ConfigManager:
    """Manages DriftGuard configuration"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "DRIFTGUARD_"
    ):
        self.env_prefix = env_prefix
        self._config: Optional[MonitoringConfig] = None
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.load_config(config_path)
        else:
            self._load_from_environment()
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            self._config = MonitoringConfig(**config_dict)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        config_dict = {
            "project_name": os.getenv(f"{self.env_prefix}PROJECT_NAME", "default"),
            "model_name": os.getenv(f"{self.env_prefix}MODEL_NAME", "model"),
            "version": os.getenv(f"{self.env_prefix}VERSION", "0.1.0"),
            "drift": {
                "method": os.getenv(f"{self.env_prefix}DRIFT_METHOD", "ks_test"),
                "threshold": float(os.getenv(f"{self.env_prefix}DRIFT_THRESHOLD", "0.05")),
                "window_size": int(os.getenv(f"{self.env_prefix}WINDOW_SIZE", "1000")),
            },
            "metrics": [
                {
                    "name": "accuracy",
                    "threshold": 0.9,
                    "comparison_window": 1000
                }
            ],
            "alerts": {
                "email_recipients": [
                    email.strip() for email in 
                    os.getenv(f"{self.env_prefix}ALERT_EMAILS", "").split(",")
                    if email.strip()
                ],
                "cooldown_period": int(os.getenv(f"{self.env_prefix}ALERT_COOLDOWN", "3600")),
            },
            "storage": {
                "storage_type": os.getenv(f"{self.env_prefix}STORAGE_TYPE", "local"),
                "path": os.getenv(f"{self.env_prefix}STORAGE_PATH", "./monitoring_data"),
            }
        }
        
        try:
            self._config = MonitoringConfig(**config_dict)
            self.logger.info("Loaded configuration from environment")
        except Exception as e:
            self.logger.error(f"Failed to load config from environment: {str(e)}")
            raise
    
    @property
    def config(self) -> MonitoringConfig:
        """Get the current configuration"""
        if self._config is None:
            self._load_from_environment()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        if self._config is None:
            self._load_from_environment()
            
        config_dict = self._config.dict()
        config_dict.update(updates)
        
        try:
            self._config = MonitoringConfig(**config_dict)
            self.logger.info("Updated configuration")
        except Exception as e:
            self.logger.error(f"Failed to update config: {str(e)}")
            raise
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration"""
        if self._config is None:
            return ["No configuration loaded"]
            
        issues = []
        
        # Check storage path
        storage = self._config.storage
        if storage.storage_type == "local":
            path = Path(storage.path)
            if not path.exists():
                try:
                    path.mkdir(parents=True)
                except Exception as e:
                    issues.append(f"Cannot create storage directory: {str(e)}")
        
        # Check alert configuration
        alerts = self._config.alerts
        if not alerts.email_recipients and not alerts.slack_webhook:
            issues.append("No alert destinations configured")
        
        # Check metric configurations
        if not self._config.metrics:
            issues.append("No metrics configured")
        
        return issues
    
    def get_storage_client(self):
        """Get appropriate storage client based on configuration"""
        storage_type = self.config.storage.storage_type
        
        if storage_type == "local":
            return LocalStorageClient(self.config.storage)
        elif storage_type == "s3":
            return S3StorageClient(self.config.storage)
        elif storage_type == "gcs":
            return GCSStorageClient(self.config.storage)
        elif storage_type == "azure":
            return AzureStorageClient(self.config.storage)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

class LocalStorageClient:
    """Client for local file storage"""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.path)
        self.base_path.mkdir(parents=True, exist_ok=True)

class S3StorageClient:
    """Client for AWS S3 storage"""
    def __init__(self, config: StorageConfig):
        # Implementation for S3
        pass

class GCSStorageClient:
    """Client for Google Cloud Storage"""
    def __init__(self, config: StorageConfig):
        # Implementation for GCS
        pass

class AzureStorageClient:
    """Client for Azure Blob Storage"""
    def __init__(self, config: StorageConfig):
        # Implementation for Azure
        pass
