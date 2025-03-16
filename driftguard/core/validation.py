"""
Data validation module for DriftGuard.
"""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pydantic import BaseModel, Field, validator
from .interfaces import IDataValidator

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    """Validation result model"""
    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistics for each feature"
    )

class DataValidator(IDataValidator):
    """Data validation for model inputs"""
    
    def __init__(
        self,
        schema: Optional[Dict] = None,
        required_columns: Optional[List[str]] = None,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        categorical_values: Optional[Dict[str, List[Union[str, int]]]] = None,
        max_missing_pct: float = 0.1
    ):
        """Initialize validator"""
        self.schema = schema or {}
        self.required_columns = required_columns or []
        self.value_ranges = value_ranges or {}
        self.categorical_values = categorical_values or {}
        self.max_missing_pct = max_missing_pct
        self.reference_stats = {}
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        try:
            # Compute reference statistics
            self.reference_stats = self._compute_statistics(reference_data)
            
            # Infer schema if not provided
            if not self.schema:
                self.schema = self._infer_schema(reference_data)
            
            # Infer required columns if not provided
            if not self.required_columns:
                self.required_columns = list(reference_data.columns)
            
            # Infer value ranges if not provided
            if not self.value_ranges:
                self.value_ranges = self._infer_ranges(reference_data)
            
            # Infer categorical values if not provided
            if not self.categorical_values:
                self.categorical_values = self._infer_categories(reference_data)
            
            logger.info("Initialized data validator")
            
        except Exception as e:
            logger.error(f"Failed to initialize validator: {str(e)}")
            raise
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for each feature"""
        stats = {}
        for col in data.columns:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'missing_pct': float(data[col].isnull().mean())
            }
        return stats
    
    def _infer_schema(self, data: pd.DataFrame) -> Dict:
        """Infer data schema from reference data"""
        schema = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                schema[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                schema[col] = 'datetime'
            elif pd.api.types.is_categorical_dtype(data[col]):
                schema[col] = 'categorical'
            else:
                schema[col] = 'string'
        return schema
    
    def _infer_ranges(
        self,
        data: pd.DataFrame,
        std_multiplier: float = 3
    ) -> Dict[str, Tuple[float, float]]:
        """Infer acceptable value ranges"""
        ranges = {}
        for col in data.columns:
            if self.schema.get(col) == 'numeric':
                mean = data[col].mean()
                std = data[col].std()
                ranges[col] = (
                    float(mean - std_multiplier * std),
                    float(mean + std_multiplier * std)
                )
        return ranges
    
    def _infer_categories(
        self,
        data: pd.DataFrame,
        max_unique: int = 100
    ) -> Dict[str, List[Union[str, int]]]:
        """Infer categorical values"""
        categories = {}
        for col in data.columns:
            if self.schema.get(col) == 'categorical':
                unique_values = data[col].unique()
                if len(unique_values) <= max_unique:
                    categories[col] = list(unique_values)
        return categories
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate input data"""
        try:
            errors = []
            warnings = []
            
            # Check required columns
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                errors.append(
                    f"Missing required columns: {', '.join(missing_cols)}"
                )
            
            # Compute current statistics
            current_stats = self._compute_statistics(data)
            
            for col in data.columns:
                # Check data type
                expected_type = self.schema.get(col)
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column {col} should be numeric")
                
                # Check missing values
                missing_pct = data[col].isnull().mean()
                if missing_pct > self.max_missing_pct:
                    errors.append(
                        f"Column {col} has {missing_pct:.1%} missing values"
                        f" (max allowed: {self.max_missing_pct:.1%})"
                    )
                
                # Check value ranges
                if col in self.value_ranges:
                    min_val, max_val = self.value_ranges[col]
                    if data[col].min() < min_val or data[col].max() > max_val:
                        warnings.append(
                            f"Column {col} has values outside expected range"
                            f" [{min_val:.2f}, {max_val:.2f}]"
                        )
                
                # Check categorical values
                if col in self.categorical_values:
                    invalid_values = set(data[col].unique()) - set(self.categorical_values[col])
                    if invalid_values:
                        warnings.append(
                            f"Column {col} has unexpected categories: "
                            f"{', '.join(map(str, invalid_values))}"
                        )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                stats=current_stats
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data"""
        try:
            # Update reference statistics
            self.reference_stats = self._compute_statistics(new_reference)
            
            # Update schema and ranges
            self.schema = self._infer_schema(new_reference)
            self.value_ranges = self._infer_ranges(new_reference)
            self.categorical_values = self._infer_categories(new_reference)
            
            logger.info("Updated reference data")
            
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")
            raise
