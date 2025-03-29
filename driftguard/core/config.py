"""
Enhanced configuration module for DriftGuard v0.1.5.
Includes advanced monitoring, alerting, and ML options.
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, EmailStr, Field, model_validator, field_validator, validator
from datetime import timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MLConfig(BaseModel):
    """Machine learning configuration settings"""
    feature_importance: bool = True
    shap_backend: str = "auto"
    n_jobs: int = -1  # -1 means use all cores
    batch_size: int = 1000
    cache_dir: Optional[str] = None
    
    @validator('n_jobs')
    def validate_n_jobs(cls, v):
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        if v != -1 and (v < 1 or v > max_cores):
            raise ValueError(f"n_jobs must be -1 or between 1 and {max_cores}")
        return v

class EmailConfig(BaseModel):
    """Enhanced email configuration settings"""
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    default_recipients: List[EmailStr] = []
    use_ssl: bool = False
    use_tls: bool = True
    timeout: int = 30
    retry_count: int = 3
    template_dir: Optional[str] = None
    
    @validator('smtp_port')
    def validate_port(cls, v):
        if not 0 <= v <= 65535:
            raise ValueError("SMTP port must be between 0 and 65535")
        return v
    
    @model_validator(mode='after')
    def validate_security(cls, values):
        if values.get('use_ssl') and values.get('use_tls'):
            raise ValueError("Cannot use both SSL and TLS")
        return values
    
    @model_validator(mode='after')
    def check_credentials(cls, values):
        """Check if email credentials are provided or in environment."""
        username = values.smtp_user or os.getenv("DRIFTGUARD_EMAIL_USER")
        password = values.smtp_password or os.getenv("DRIFTGUARD_EMAIL_PASSWORD")
        
        if not username or not password:
            raise ValueError(
                "Email credentials must be provided either in config "
                "or as environment variables DRIFTGUARD_EMAIL_USER and DRIFTGUARD_EMAIL_PASSWORD"
            )
        
        values.smtp_user = username
        values.smtp_password = password
        return values

class SlackConfig(BaseModel):
    """Slack integration configuration"""
    enabled: bool = False
    webhook_url: Optional[str] = None
    channel: Optional[str] = None
    username: str = "DriftGuard"
    icon_emoji: str = ":robot_face:"
    
    @validator('webhook_url')
    def validate_webhook(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v

class AlertConfig(BaseModel):
    """Enhanced alert configuration settings"""
    severity_levels: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    notify_on_severity: List[str] = ["ERROR", "CRITICAL"]
    email: Optional[EmailConfig] = None
    slack: Optional[SlackConfig] = None
    alert_cooldown: timedelta = timedelta(hours=1)
    max_alerts_per_day: int = 100
    aggregation_window: timedelta = timedelta(minutes=5)
    
    schema: Dict = Field(default_factory=lambda: {
        "validate": True,
        "allow_extra": False,
        "strict_types": True
    })
    
    missing: Dict = Field(default_factory=lambda: {
        "max_pct": 0.1,
        "imputation_strategy": "mean",
        "categorical_strategy": "mode"
    })
    
    range: Dict = Field(default_factory=lambda: {
        "validate": True,
        "std_threshold": 3.0,
        "quantile_range": [0.001, 0.999]
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
    
    @model_validator(mode='after')
    def check_email_config(cls, values):
        """Check if email config is present when email is set."""
        if values.email and not values.email.smtp_user:
            values.email.smtp_user = os.getenv("DRIFTGUARD_EMAIL_USER")
        if values.email and not values.email.smtp_password:
            values.email.smtp_password = os.getenv("DRIFTGUARD_EMAIL_PASSWORD")
        return values

class DriftConfig(BaseModel):
    """Enhanced drift detection configuration"""
    methods: List[str] = [
        "ks", "jsd", "psi", "wasserstein", 
        "chi2", "anderson"
    ]
    thresholds: Dict[str, float] = {
        "ks": 0.05,
        "jsd": 0.1,
        "psi": 0.2,
        "wasserstein": 0.1,
        "chi2": 0.05,
        "anderson": 0.05
    }
    window_size: int = 1000
    min_samples: int = 100
    feature_subset_size: Optional[int] = None
    sampling_strategy: str = "random"
    confidence_level: float = 0.95
    correction_method: str = "bonferroni"
    
    @validator('methods')
    def validate_methods(cls, v):
        valid = ["ks", "jsd", "psi", "wasserstein", "chi2", "anderson"]
        for method in v:
            if method not in valid:
                raise ValueError(
                    f"Invalid method '{method}'. "
                    f"Must be one of: {', '.join(valid)}"
                )
        return v
    
    @validator('thresholds')
    def validate_thresholds(cls, v, values):
        methods = values.get('methods', [])
        for method in methods:
            if method not in v:
                raise ValueError(f"Missing threshold for method: {method}")
        return v
    
    @validator('window_size', 'min_samples')
    def validate_window(cls, v):
        if v < 10:
            raise ValueError("Window size must be at least 10")
        return v
    
    @validator('confidence_level')
    def validate_confidence(cls, v):
        if not 0 < v < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        return v
    
    @field_validator("methods")
    def validate_drift_config(cls, v):
        """Validate drift detection configuration."""
        valid_methods = {"ks", "anderson", "wasserstein"}
        for method in v:
            if method not in valid_methods:
                raise ValueError(f"Invalid drift detection method: {method}")
            if method not in cls.thresholds:
                raise ValueError(f"Missing threshold for method: {method}")
        return v

class MonitorConfig(BaseModel):
    """Enhanced model monitoring configuration"""
    metrics: List[str] = [
        "accuracy", "f1", "precision", "recall",
        "roc_auc", "pr_auc", "mcc", "brier", "log_loss"
    ]
    threshold_type: str = "dynamic"  # absolute, relative, or dynamic
    thresholds: Dict[str, float] = {
        "accuracy": 0.8,
        "f1": 0.7,
        "precision": 0.7,
        "recall": 0.7,
        "roc_auc": 0.75,
        "pr_auc": 0.7,
        "mcc": 0.5,
        "brier": 0.2,
        "log_loss": 0.5
    }
    window_size: int = 1000
    history_size: int = 100
    degradation_threshold: float = 0.1
    min_retrain_days: int = 7
    retrain_threshold: int = 3
    feature_tracking: bool = True
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid = [
            "accuracy", "f1", "precision", "recall",
            "roc_auc", "pr_auc", "mcc", "brier", "log_loss"
        ]
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
        for metric in metrics:
            if metric not in v:
                raise ValueError(f"Missing threshold for metric: {metric}")
        return v
    
    @validator('window_size', 'history_size')
    def validate_sizes(cls, v):
        if v < 10:
            raise ValueError("Size must be at least 10")
        return v
    
    @field_validator("metrics")
    def validate_monitor_config(cls, v):
        """Validate monitoring configuration."""
        valid_metrics = {
            "accuracy", "precision", "recall", "f1", 
            "roc_auc", "log_loss", "brier_score"
        }
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}")
        return v

class StorageConfig(BaseModel):
    """Storage configuration settings"""
    path: str = "./storage"
    format: str = "parquet"
    compression: str = "snappy"
    partition_by: Optional[List[str]] = None
    max_size_gb: float = 10.0
    cleanup_age_days: int = 30
    
    @validator('path')
    def validate_path(cls, v):
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
        return v
    
    @validator('format')
    def validate_format(cls, v):
        valid = ["parquet", "csv", "json"]
        if v not in valid:
            raise ValueError(f"Format must be one of: {', '.join(valid)}")
        return v

class Config(BaseModel):
    """Enhanced main configuration for DriftGuard"""
    alerts: AlertConfig = AlertConfig()
    drift: DriftConfig = DriftConfig()
    monitor: MonitorConfig = MonitorConfig()
    ml: MLConfig = MLConfig()
    storage: StorageConfig = StorageConfig()
    log_level: str = "INFO"
    environment: str = "production"
    version: str = "0.1.5"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        valid = ["development", "staging", "production"]
        if v not in valid:
            raise ValueError(
                f"Invalid environment '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"
