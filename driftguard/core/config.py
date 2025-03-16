"""
Configuration module for DriftGuard.
"""
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from .interfaces import DriftMethod, AlertSeverity

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    ks_test: Dict[str, Union[float, str]] = Field(
        default={
            "threshold": 0.05,
            "correction": "bonferroni"
        }
    )
    jsd: Dict[str, Union[float, int]] = Field(
        default={
            "threshold": 0.1,
            "bins": 20
        }
    )
    psi: Dict[str, Union[float, int]] = Field(
        default={
            "threshold": 0.2,
            "bins": 20
        }
    )
    feature_selection: Dict[str, Optional[Union[str, int]]] = Field(
        default={
            "method": "all",
            "max_features": None
        }
    )

class MonitorConfig(BaseModel):
    """Model monitoring configuration"""
    degradation_threshold: float = 0.1
    window_size: int = 100
    metrics: Dict[str, List[str]] = Field(
        default={
            "classification": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ],
            "regression": [
                "mse",
                "rmse",
                "mae",
                "r2",
                "explained_variance"
            ]
        }
    )

class ValidationConfig(BaseModel):
    """Data validation configuration"""
    schema: Dict[str, bool] = Field(
        default={
            "validate": True,
            "allow_extra": False
        }
    )
    missing: Dict[str, Union[float, str]] = Field(
        default={
            "max_pct": 0.1,
            "strategy": "drop"
        }
    )
    range: Dict[str, Union[bool, float]] = Field(
        default={
            "validate": True,
            "std_threshold": 3.0
        }
    )

class EmailConfig(BaseModel):
    """Email configuration for alerts"""
    enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 25
    from_address: str = ""
    to_addresses: List[str] = []

class AlertConfig(BaseModel):
    """Alert management configuration"""
    enabled: bool = True
    severity_levels: List[AlertSeverity] = Field(
        default=[
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]
    )
    email: EmailConfig = EmailConfig()

class StorageConfig(BaseModel):
    """Storage configuration"""
    path: str = "storage"
    retention_days: int = 7
    compression: bool = True

class DriftGuardConfig(BaseModel):
    """Main configuration for DriftGuard"""
    drift: DriftConfig = DriftConfig()
    monitor: MonitorConfig = MonitorConfig()
    validation: ValidationConfig = ValidationConfig()
    alerts: AlertConfig = AlertConfig()
    storage: StorageConfig = StorageConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> "DriftGuardConfig":
        """Load configuration from YAML file"""
        import yaml
        
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
