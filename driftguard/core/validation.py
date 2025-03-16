"""
Data validation module for DriftGuard.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Dict[str, float]]

class DataValidator:
    """Validates input data for consistency and quality"""
    
    def __init__(self, max_missing_pct: float = 0.1):
        """Initialize validator"""
        self.max_missing_pct = max_missing_pct
        self.reference_schema = None
        self.feature_ranges = {}
        self.feature_types = {}
        self._initialized = False
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        self.reference_schema = reference_data.columns.tolist()
        
        # Store feature types
        self.feature_types = {
            col: str(reference_data[col].dtype)
            for col in reference_data.columns
        }
        
        # Compute feature ranges
        self.feature_ranges = {}
        for col in reference_data.columns:
            if np.issubdtype(reference_data[col].dtype, np.number):
                values = reference_data[col].dropna()
                if len(values) > 0:
                    self.feature_ranges[col] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
        
        self._initialized = True
    
    def _validate_schema(self, data: pd.DataFrame) -> List[str]:
        """Validate data schema"""
        errors = []
        
        # Check for missing columns
        missing_cols = set(self.reference_schema) - set(data.columns)
        if missing_cols:
            errors.append(
                f"Missing columns: {', '.join(missing_cols)}"
            )
        
        # Check for extra columns
        extra_cols = set(data.columns) - set(self.reference_schema)
        if extra_cols:
            errors.append(
                f"Extra columns found: {', '.join(extra_cols)}"
            )
        
        # Check data types
        for col in self.reference_schema:
            if col in data.columns:
                expected_type = self.feature_types[col]
                actual_type = str(data[col].dtype)
                if expected_type != actual_type:
                    errors.append(
                        f"Column '{col}' has incorrect type. "
                        f"Expected {expected_type}, got {actual_type}"
                    )
        
        return errors
    
    def _validate_missing_values(self, data: pd.DataFrame) -> List[str]:
        """Validate missing values"""
        errors = []
        
        for col in data.columns:
            missing_pct = data[col].isna().mean()
            if missing_pct > self.max_missing_pct:
                errors.append(
                    f"Column '{col}' has {missing_pct:.1%} missing values, "
                    f"exceeding threshold of {self.max_missing_pct:.1%}"
                )
        
        return errors
    
    def _validate_ranges(self, data: pd.DataFrame) -> List[str]:
        """Validate numerical ranges"""
        warnings = []
        
        for col, ranges in self.feature_ranges.items():
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            # Check for values outside 3 standard deviations
            mean = ranges['mean']
            std = ranges['std']
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(values)
                warnings.append(
                    f"Column '{col}' has {outlier_pct:.1%} values outside "
                    f"3 standard deviations from the mean"
                )
            
            # Check for values outside absolute ranges
            values_below = values[values < ranges['min']]
            values_above = values[values > ranges['max']]
            
            if len(values_below) > 0:
                pct_below = len(values_below) / len(values)
                warnings.append(
                    f"Column '{col}' has {pct_below:.1%} values below "
                    f"minimum reference value {ranges['min']:.2f}"
                )
            
            if len(values_above) > 0:
                pct_above = len(values_above) / len(values)
                warnings.append(
                    f"Column '{col}' has {pct_above:.1%} values above "
                    f"maximum reference value {ranges['max']:.2f}"
                )
        
        return warnings
    
    def _compute_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute data statistics"""
        stats = {}
        
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                values = data[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'missing_pct': float(data[col].isna().mean())
                    }
        
        return stats
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate input data.
        
        Performs the following checks:
        1. Schema validation (columns and types)
        2. Missing value checks
        3. Range validation for numerical features
        """
        if not self._initialized:
            raise ValueError("Validator not initialized")
        
        errors = []
        warnings = []
        
        # Schema validation
        errors.extend(self._validate_schema(data))
        
        # Only proceed with other checks if schema is valid
        if not errors:
            # Missing value validation
            errors.extend(self._validate_missing_values(data))
            
            # Range validation
            warnings.extend(self._validate_ranges(data))
        
        # Compute statistics
        stats = self._compute_stats(data)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        self.initialize(new_reference)
