"""
Data validation module for DriftGuard.
Ensures data quality and consistency before processing.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality and consistency"""
    
    def __init__(self):
        """Initialize validator"""
        self.schema = None
        self.feature_ranges = {}
        self.feature_types = {}
        self.missing_thresholds = {}
    
    def initialize(self, reference_data: pd.DataFrame) -> None:
        """Initialize validator with reference data"""
        try:
            # Store schema information
            self.schema = {
                'columns': list(reference_data.columns),
                'dtypes': reference_data.dtypes.to_dict()
            }
            
            # Calculate feature ranges for numerical columns
            for col in reference_data.select_dtypes(include=['int64', 'float64']).columns:
                self.feature_ranges[col] = {
                    'min': float(reference_data[col].min()),
                    'max': float(reference_data[col].max()),
                    'mean': float(reference_data[col].mean()),
                    'std': float(reference_data[col].std())
                }
            
            # Store feature types
            self.feature_types = {
                col: str(dtype)
                for col, dtype in reference_data.dtypes.items()
            }
            
            # Calculate missing value thresholds
            missing_rates = reference_data.isnull().mean()
            self.missing_thresholds = {
                col: rate + 0.1  # Allow 10% more missing values
                for col, rate in missing_rates.items()
            }
            
            logger.info("Initialized data validator")
            
        except Exception as e:
            logger.error(f"Failed to initialize validator: {str(e)}")
            raise
    
    def validate_schema(self, data: pd.DataFrame) -> List[str]:
        """Validate data schema"""
        if self.schema is None:
            raise ValueError("Validator not initialized")
        
        errors = []
        
        # Check columns
        missing_cols = set(self.schema['columns']) - set(data.columns)
        extra_cols = set(data.columns) - set(self.schema['columns'])
        
        if missing_cols:
            errors.append(
                f"Missing columns: {', '.join(missing_cols)}"
            )
        if extra_cols:
            errors.append(
                f"Extra columns: {', '.join(extra_cols)}"
            )
        
        # Check data types
        for col in set(self.schema['columns']) & set(data.columns):
            expected_type = self.schema['dtypes'][col]
            actual_type = data[col].dtype
            if str(actual_type) != str(expected_type):
                errors.append(
                    f"Column '{col}' has incorrect type: "
                    f"expected {expected_type}, got {actual_type}"
                )
        
        return errors
    
    def validate_ranges(self, data: pd.DataFrame) -> List[str]:
        """Validate numerical ranges"""
        if self.schema is None:
            raise ValueError("Validator not initialized")
        
        errors = []
        
        for col, ranges in self.feature_ranges.items():
            if col not in data.columns:
                continue
            
            col_data = data[col]
            
            # Check for values outside expected range
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            mean_val = float(col_data.mean())
            std_val = float(col_data.std())
            
            # Allow values within 3 standard deviations
            expected_min = ranges['min'] - 3 * ranges['std']
            expected_max = ranges['max'] + 3 * ranges['std']
            
            if min_val < expected_min:
                errors.append(
                    f"Column '{col}' has values below expected range: "
                    f"min={min_val:.2f}, expected >={expected_min:.2f}"
                )
            
            if max_val > expected_max:
                errors.append(
                    f"Column '{col}' has values above expected range: "
                    f"max={max_val:.2f}, expected <={expected_max:.2f}"
                )
            
            # Check for significant distribution shifts
            if abs(mean_val - ranges['mean']) > 2 * ranges['std']:
                errors.append(
                    f"Column '{col}' shows significant distribution shift: "
                    f"mean={mean_val:.2f}, expected={ranges['mean']:.2f}"
                )
        
        return errors
    
    def validate_missing(self, data: pd.DataFrame) -> List[str]:
        """Validate missing values"""
        if self.schema is None:
            raise ValueError("Validator not initialized")
        
        errors = []
        
        # Check missing rates
        missing_rates = data.isnull().mean()
        for col, rate in missing_rates.items():
            if col not in self.missing_thresholds:
                continue
            
            threshold = self.missing_thresholds[col]
            if rate > threshold:
                errors.append(
                    f"Column '{col}' has too many missing values: "
                    f"{rate*100:.1f}%, threshold={threshold*100:.1f}%"
                )
        
        return errors
    
    def validate_types(self, data: pd.DataFrame) -> List[str]:
        """Validate data types"""
        if self.schema is None:
            raise ValueError("Validator not initialized")
        
        errors = []
        
        for col, expected_type in self.feature_types.items():
            if col not in data.columns:
                continue
            
            # Check if data can be converted to expected type
            try:
                if expected_type in ['int64', 'float64']:
                    pd.to_numeric(data[col])
                elif expected_type == 'bool':
                    data[col].astype(bool)
                elif expected_type == 'datetime64[ns]':
                    pd.to_datetime(data[col])
            except Exception as e:
                errors.append(
                    f"Column '{col}' contains invalid values for type {expected_type}: "
                    f"{str(e)}"
                )
        
        return errors
    
    def validate_all(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Run all validations"""
        all_errors = []
        
        try:
            # Run all validations
            schema_errors = self.validate_schema(data)
            range_errors = self.validate_ranges(data)
            missing_errors = self.validate_missing(data)
            type_errors = self.validate_types(data)
            
            # Combine all errors
            all_errors.extend(schema_errors)
            all_errors.extend(range_errors)
            all_errors.extend(missing_errors)
            all_errors.extend(type_errors)
            
            return len(all_errors) == 0, all_errors
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            all_errors.append(f"Validation error: {str(e)}")
            return False, all_errors
