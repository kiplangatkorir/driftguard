"""
Data validation system for DriftGuard.
Ensures data quality and consistency before processing.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from .interfaces import IDataValidator

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Definition of a data validation rule"""
    name: str
    condition: callable
    message: str
    severity: str = "error"  # error, warning
    
    def check(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Apply validation rule to data"""
        try:
            result = self.condition(data)
            return result, None if result else self.message
        except Exception as e:
            return False, f"{self.message} (Error: {str(e)})"

class DataSchema:
    """Schema definition for data validation"""
    
    def __init__(
        self,
        feature_dtypes: Dict[str, str],
        required_features: Optional[List[str]] = None,
        allow_extra_features: bool = False,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        min_rows: int = 1,
        max_rows: Optional[int] = None
    ):
        self.feature_dtypes = feature_dtypes
        self.required_features = required_features or list(feature_dtypes.keys())
        self.allow_extra_features = allow_extra_features
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.min_rows = min_rows
        self.max_rows = max_rows
        
        # Validate schema configuration
        self._validate_schema_config()
    
    def _validate_schema_config(self) -> None:
        """Validate schema configuration"""
        # Check feature lists consistency
        all_features = set(self.feature_dtypes.keys())
        if not all(f in all_features for f in self.required_features):
            raise ValueError("Required features must be subset of feature_dtypes")
            
        if self.categorical_features and not all(f in all_features for f in self.categorical_features):
            raise ValueError("Categorical features must be subset of feature_dtypes")
            
        if self.numerical_features and not all(f in all_features for f in self.numerical_features):
            raise ValueError("Numerical features must be subset of feature_dtypes")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary"""
        return {
            "feature_dtypes": self.feature_dtypes,
            "required_features": self.required_features,
            "allow_extra_features": self.allow_extra_features,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows
        }
    
    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> 'DataSchema':
        """Create schema from dictionary"""
        return cls(**schema_dict)
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        required_features: Optional[List[str]] = None,
        categorical_threshold: int = 10
    ) -> 'DataSchema':
        """Create schema from DataFrame"""
        feature_dtypes = {}
        categorical_features = []
        numerical_features = []
        
        for column in df.columns:
            dtype = str(df[column].dtype)
            feature_dtypes[column] = dtype
            
            # Infer categorical features
            if df[column].dtype == 'object' or (df[column].nunique() <= categorical_threshold):
                categorical_features.append(column)
            elif np.issubdtype(df[column].dtype, np.number):
                numerical_features.append(column)
        
        return cls(
            feature_dtypes=feature_dtypes,
            required_features=required_features or list(df.columns),
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )

class DataValidator(IDataValidator):
    """Validates data against defined schema and rules"""
    
    def __init__(
        self,
        schema: DataSchema,
        custom_rules: Optional[List[ValidationRule]] = None
    ):
        self.schema = schema
        self.custom_rules = custom_rules or []
        self._build_default_rules()
    
    def _build_default_rules(self) -> None:
        """Build default validation rules"""
        self.rules = [
            ValidationRule(
                name="required_features",
                condition=lambda df: all(f in df.columns for f in self.schema.required_features),
                message="Missing required features"
            ),
            ValidationRule(
                name="no_empty_data",
                condition=lambda df: not df.empty,
                message="Data is empty"
            ),
            ValidationRule(
                name="min_rows",
                condition=lambda df: len(df) >= self.schema.min_rows,
                message=f"Data has fewer than {self.schema.min_rows} rows"
            )
        ]
        
        if self.schema.max_rows:
            self.rules.append(
                ValidationRule(
                    name="max_rows",
                    condition=lambda df: len(df) <= self.schema.max_rows,
                    message=f"Data has more than {self.schema.max_rows} rows"
                )
            )
        
        # Add dtype validation rules
        for feature, dtype in self.schema.feature_dtypes.items():
            self.rules.append(
                ValidationRule(
                    name=f"dtype_{feature}",
                    condition=lambda df, f=feature, d=dtype: (
                        f not in df.columns or str(df[f].dtype) == d
                    ),
                    message=f"Feature {feature} has incorrect dtype"
                )
            )
        
        # Add custom rules
        self.rules.extend(self.custom_rules)
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data against schema and rules.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check extra features
        if not self.schema.allow_extra_features:
            extra_features = set(data.columns) - set(self.schema.feature_dtypes.keys())
            if extra_features:
                is_valid = False
                messages.append(f"Unexpected features found: {extra_features}")
        
        # Apply all rules
        for rule in self.rules:
            result, message = rule.check(data)
            if not result:
                if rule.severity == "error":
                    is_valid = False
                messages.append(f"{rule.severity.upper()}: {message}")
        
        # Check for nulls in required features
        for feature in self.schema.required_features:
            if feature in data.columns and data[feature].isnull().any():
                is_valid = False
                messages.append(f"Null values found in required feature: {feature}")
        
        # Additional checks for categorical features
        for feature in self.schema.categorical_features:
            if feature in data.columns:
                unique_values = data[feature].nunique()
                if unique_values > 1000:  # Warning threshold for high cardinality
                    messages.append(
                        f"WARNING: High cardinality in categorical feature {feature} "
                        f"({unique_values} unique values)"
                    )
        
        # Check for infinite values in numerical features
        for feature in self.schema.numerical_features:
            if feature in data.columns and not data[feature].dtype == 'object':
                inf_mask = np.isinf(data[feature])
                if inf_mask.any():
                    is_valid = False
                    messages.append(f"Infinite values found in feature: {feature}")
        
        return is_valid, messages
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the current schema definition"""
        return self.schema.to_dict()
    
    def validate_and_clean(
        self,
        data: pd.DataFrame,
        drop_extra_features: bool = True,
        handle_missing: str = 'raise'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean data according to schema.
        
        Args:
            data: Input DataFrame
            drop_extra_features: Whether to drop features not in schema
            handle_missing: How to handle missing values ('raise', 'drop', or 'fill')
            
        Returns:
            Tuple of (cleaned DataFrame, list of messages)
        """
        is_valid, messages = self.validate(data)
        if not is_valid and handle_missing == 'raise':
            raise ValueError("\n".join(messages))
        
        cleaned_data = data.copy()
        
        # Handle extra features
        if drop_extra_features:
            extra_features = set(cleaned_data.columns) - set(self.schema.feature_dtypes.keys())
            if extra_features:
                cleaned_data = cleaned_data.drop(columns=list(extra_features))
                messages.append(f"Dropped extra features: {extra_features}")
        
        # Handle missing values
        if handle_missing == 'drop':
            original_len = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=self.schema.required_features)
            dropped_rows = original_len - len(cleaned_data)
            if dropped_rows > 0:
                messages.append(f"Dropped {dropped_rows} rows with missing values")
        elif handle_missing == 'fill':
            for feature in cleaned_data.columns:
                if feature in self.schema.numerical_features:
                    cleaned_data[feature] = cleaned_data[feature].fillna(
                        cleaned_data[feature].mean()
                    )
                elif feature in self.schema.categorical_features:
                    cleaned_data[feature] = cleaned_data[feature].fillna(
                        cleaned_data[feature].mode()[0]
                    )
            messages.append("Filled missing values with mean/mode")
        
        return cleaned_data, messages
