"""
Configuration management for DriftGuard.
"""
from typing import Any, Dict, List, Optional, Union
import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from .interfaces import (
    AlertConfig, MetricConfig, DriftConfig,
    StorageConfig, MonitoringConfig
)

class ConfigManager:
    """Manages DriftGuard configuration"""
    
    DEFAULT_CONFIG = {
        "project_name": "driftguard_project",
        "model_type": "classification",
        "metrics": [
            {"name": "accuracy", "threshold": 0.1},
            {"name": "f1", "threshold": 0.1},
            {"name": "roc_auc", "threshold": 0.15}
        ],
        "drift": {
            "method": "ks_test",
            "threshold": 0.05,
            "window_size": 1000
        },
        "alerts": {
            "email_enabled": False,
            "slack_enabled": False
        },
        "storage": {
            "type": "local",
            "path": "./storage",
            "retention_days": 30
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> MonitoringConfig:
        """Load configuration from file or use defaults"""
        config_dict = self.DEFAULT_CONFIG.copy()
        
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                config_dict.update(file_config)
            except Exception as e:
                raise ValueError(f"Failed to load config file: {str(e)}")
        
        # Load environment variables
        self._update_from_env(config_dict)
        
        try:
            return MonitoringConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    def _update_from_env(self, config: Dict[str, Any]) -> None:
        """Update configuration with environment variables"""
        env_mapping = {
            "DRIFTGUARD_PROJECT_NAME": ("project_name", str),
            "DRIFTGUARD_MODEL_TYPE": ("model_type", str),
            "DRIFTGUARD_DRIFT_THRESHOLD": ("drift.threshold", float),
            "DRIFTGUARD_DRIFT_METHOD": ("drift.method", str),
            "DRIFTGUARD_EMAIL_ENABLED": ("alerts.email_enabled", bool),
            "DRIFTGUARD_EMAIL_SMTP_HOST": ("alerts.email_smtp_host", str),
            "DRIFTGUARD_EMAIL_SMTP_PORT": ("alerts.email_smtp_port", int),
            "DRIFTGUARD_EMAIL_SMTP_USER": ("alerts.email_smtp_user", str),
            "DRIFTGUARD_EMAIL_SMTP_PASSWORD": ("alerts.email_smtp_password", str),
            "DRIFTGUARD_SLACK_ENABLED": ("alerts.slack_enabled", bool),
            "DRIFTGUARD_SLACK_WEBHOOK": ("alerts.slack_webhook", str),
            "DRIFTGUARD_STORAGE_PATH": ("storage.path", str),
            "DRIFTGUARD_STORAGE_RETENTION": ("storage.retention_days", int)
        }
        
        for env_var, (config_path, type_func) in env_mapping.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    self._set_nested_value(config, config_path, value)
                except Exception as e:
                    raise ValueError(
                        f"Invalid environment variable {env_var}: {str(e)}"
                    )
    
    def _set_nested_value(
        self,
        config: Dict[str, Any],
        path: str,
        value: Any
    ) -> None:
        """Set value in nested dictionary using dot notation"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        config_dict = self.config.model_dump()
        
        for key, value in updates.items():
            if '.' in key:
                self._set_nested_value(config_dict, key, value)
            else:
                config_dict[key] = value
        
        try:
            self.config = MonitoringConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration update: {str(e)}")
        
        # Save to file if path exists
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    yaml.safe_dump(config_dict, f)
            except Exception as e:
                raise ValueError(f"Failed to save config file: {str(e)}")
    
    def get_config(self) -> MonitoringConfig:
        """Get current configuration"""
        return self.config
