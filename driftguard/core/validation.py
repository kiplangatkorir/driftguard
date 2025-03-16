"""
Data validation module for DriftGuard.
"""
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime

from .interfaces import IDataValidator, ValidationResult
from .config import ValidationConfig

class DataValidator(IDataValidator):
    """Validates data quality and consistency"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize data validator"""
        self.config = config or ValidationConfig()
        self.reference_schema = None
        self.reference_stats = None
        self._initialized = False
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
        
        # Store reference schema
        self.reference_schema = {
            col: str(dtype) for col, dtype in reference_data.dtypes.items()
        }
        
        # Compute reference statistics
        self.reference_stats = {}
        for column in reference_data.columns:
            if pd.api.types.is_numeric_dtype(reference_data[column]):
                stats = reference_data[column].agg(['mean', 'std']).to_dict()
                self.reference_stats[column] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': reference_data[column].min(),
                    'max': reference_data[column].max()
                }
        
        self._initialized = True
    
    def _validate_schema(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        """Validate data schema"""
        errors = []
        
        # Check for missing columns
        missing_cols = set(self.reference_schema.keys()) - set(data.columns)
        if missing_cols:
            errors.append(
                f"Missing columns: {', '.join(missing_cols)}"
            )
        
        # Check for extra columns
        if not self.config.schema["allow_extra"]:
            extra_cols = set(data.columns) - set(self.reference_schema.keys())
            if extra_cols:
                errors.append(
                    f"Extra columns not allowed: {', '.join(extra_cols)}"
                )
        
        # Check data types
        for col, ref_dtype in self.reference_schema.items():
            if col in data.columns:
                curr_dtype = str(data[col].dtype)
                if curr_dtype != ref_dtype:
                    errors.append(
                        f"Column '{col}' has incorrect type. "
                        f"Expected {ref_dtype}, got {curr_dtype}"
                    )
        
        return errors
    
    def _validate_missing_values(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        """Validate missing values"""
        errors = []
        max_missing_pct = self.config.missing["max_pct"]
        
        for column in self.reference_schema.keys():
            if column in data.columns:
                missing_pct = data[column].isna().mean()
                if missing_pct > max_missing_pct:
                    errors.append(
                        f"Column '{column}' has {missing_pct:.1%} missing values, "
                        f"exceeding threshold of {max_missing_pct:.1%}"
                    )
        
        return errors
    
    def _validate_ranges(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        """Validate numeric ranges"""
        errors = []
        std_threshold = self.config.range["std_threshold"]
        
        for column, stats in self.reference_stats.items():
            if column not in data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            
            # Check for values outside expected range
            mean, std = stats['mean'], stats['std']
            lower_bound = mean - std_threshold * std
            upper_bound = mean + std_threshold * std
            
            outliers = data[
                (data[column] < lower_bound) |
                (data[column] > upper_bound)
            ]
            
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(data)
                errors.append(
                    f"Column '{column}' has {outlier_pct:.1%} values "
                    f"outside expected range [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
        
        return errors
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data quality"""
        if not self._initialized:
            raise ValueError("Validator not initialized")
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                errors=["Data cannot be empty"],
                warnings=[]
            )
        
        errors = []
        warnings = []
        
        # Schema validation
        if self.config.schema["validate"]:
            schema_errors = self._validate_schema(data)
            errors.extend(schema_errors)
        
        # Missing value validation
        missing_errors = self._validate_missing_values(data)
        errors.extend(missing_errors)
        
        # Range validation
        if self.config.range["validate"]:
            range_errors = self._validate_ranges(data)
            warnings.extend(range_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
