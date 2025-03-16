"""
Configuration module for DriftGuard.
"""
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class DriftMethod(str, Enum):
    """Drift detection methods"""
    KS_TEST = "ks_test"
    JSD = "jsd"
    PSI = "psi"

class ModelConfig(BaseModel):
    """Model configuration"""
    type: str = Field(
        description="Model type (classification or regression)"
    )
    metrics: List[str] = Field(
        description="List of metrics to track"
    )
    max_missing_pct: float = Field(
        default=0.1,
        description="Maximum allowed percentage of missing values"
    )
    threshold: float = Field(
        default=0.1,
        description="Threshold for metric degradation"
    )

class DriftConfig(BaseModel):
    """Drift detection configuration"""
    method: DriftMethod = Field(
        description="Drift detection method"
    )
    threshold: float = Field(
        default=0.05,
        description="P-value threshold for drift detection"
    )
    window_size: int = Field(
        default=1000,
        description="Window size for drift detection"
    )
    min_samples: int = Field(
        default=100,
        description="Minimum number of samples required"
    )
