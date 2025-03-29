"""
Configuration module for DriftGuard.
"""
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, EmailStr, Field, validator

class EmailConfig(BaseModel):
    """Email configuration settings"""
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    default_recipients: List[EmailStr] = []
    
    @validator('smtp_port')
    def validate_port(cls, v):
        if not 0 <= v <= 65535:
            raise ValueError("SMTP port must be between 0 and 65535")
        return v

class AlertConfig(BaseModel):
    """Alert configuration settings"""
    severity_levels: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    notify_on_severity: List[str] = ["ERROR", "CRITICAL"]
    email: Optional[EmailConfig] = None
    
    schema: Dict = Field(default_factory=lambda: {
        "validate": True,
        "allow_extra": False
    })
    
    missing: Dict = Field(default_factory=lambda: {
        "max_pct": 0.1
    })
    
    range: Dict = Field(default_factory=lambda: {
        "validate": True,
        "std_threshold": 3.0
    })
    
    @validator('severity_levels', 'notify_on_severity')
    def validate_severity(cls, v):
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in v:
            if level not in valid:
                raise ValueError(
                    f"Invalid severity level '{level}'. "
                    f"Must be one of: {', '.join(valid)}"
                )
        return v
    
    @validator('notify_on_severity')
    def validate_notify_severity(cls, v, values):
        if not set(v).issubset(set(values.get('severity_levels', []))):
            raise ValueError(
                "notify_on_severity must be a subset of severity_levels"
            )
        return v

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    methods: List[str] = ["ks", "jsd", "psi"]
    thresholds: Dict[str, float] = {
        "ks": 0.1,
        "jsd": 0.1,
        "psi": 0.2
    }
    window_size: int = 1000
    
    @validator('methods')
    def validate_methods(cls, v):
        valid = ["ks", "jsd", "psi"]
        for method in v:
            if method not in valid:
                raise ValueError(
                    f"Invalid drift detection method '{method}'. "
                    f"Must be one of: {', '.join(valid)}"
                )
        return v
    
    @validator('thresholds')
    def validate_thresholds(cls, v, values):
        methods = values.get('methods', [])
        if not set(v.keys()).issubset(set(methods)):
            raise ValueError(
                "Threshold keys must match selected methods"
            )
        return v
    
    @validator('window_size')
    def validate_window(cls, v):
        if v < 100:
            raise ValueError("Window size must be at least 100")
        return v

class MonitorConfig(BaseModel):
    """Model monitoring configuration"""
    metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
    threshold_type: str = "absolute"
    thresholds: Dict[str, float] = {
        "accuracy": 0.8,
        "f1": 0.7,
        "precision": 0.7,
        "recall": 0.7
    }
    window_size: int = 1000
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        for metric in v:
            if metric not in valid:
                raise ValueError(
                    f"Invalid metric '{metric}'. "
                    f"Must be one of: {', '.join(valid)}"
                )
        return v
    
    @validator('threshold_type')
    def validate_threshold_type(cls, v):
        valid = ["absolute", "relative", "dynamic"]
        if v not in valid:
            raise ValueError(
                f"Invalid threshold type '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v
    
    @validator('thresholds')
    def validate_thresholds(cls, v, values):
        metrics = values.get('metrics', [])
        if not set(v.keys()).issubset(set(metrics)):
            raise ValueError(
                "Threshold keys must match selected metrics"
            )
        return v

class Config(BaseModel):
    """Main configuration for DriftGuard"""
    alerts: AlertConfig = AlertConfig()
    drift: DriftConfig = DriftConfig()
    monitor: MonitorConfig = MonitorConfig()
    storage_path: str = "./storage"
    log_level: str = "INFO"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v
