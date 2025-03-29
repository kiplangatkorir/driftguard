"""Data validation and preprocessing utilities."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

def validate_inputs(
    data: Union[pd.DataFrame, np.ndarray],
    labels: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Validate and preprocess input data and labels.
    
    Args:
        data: Input features
        labels: Optional target labels
        
    Returns:
        Tuple of validated DataFrame and Series
    """
    # Convert numpy arrays to pandas
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    if labels is not None:
        if isinstance(labels, np.ndarray):
            labels = pd.Series(labels)
        
        # Check lengths match
        if len(data) != len(labels):
            raise ValueError(
                f"Data and labels must have same length. "
                f"Got {len(data)} and {len(labels)}"
            )
    
    return data, labels

def check_data_quality(data: pd.DataFrame) -> dict:
    """
    Check data quality metrics.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        "missing_values": data.isnull().sum().to_dict(),
        "unique_counts": data.nunique().to_dict(),
        "data_types": data.dtypes.astype(str).to_dict()
    }
    
    # Add basic statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics.update({
            "numeric_stats": {
                col: {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max()
                }
                for col in numeric_cols
            }
        })
    
    return metrics

def detect_categorical_features(data: pd.DataFrame) -> list:
    """
    Detect categorical features in a DataFrame.
    
    Args:
        data: Input DataFrame
        
    Returns:
        List of categorical column names
    """
    categorical = []
    
    for col in data.columns:
        # Check if string or object type
        if data[col].dtype == "O":
            categorical.append(col)
            continue
        
        # Check if numeric but few unique values
        if data[col].dtype in ["int64", "float64"]:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.05:  # Less than 5% unique values
                categorical.append(col)
    
    return categorical

def preprocess_features(
    data: pd.DataFrame,
    categorical_features: Optional[list] = None
) -> pd.DataFrame:
    """
    Preprocess features for drift detection.
    
    Args:
        data: Input DataFrame
        categorical_features: Optional list of categorical features
        
    Returns:
        Preprocessed DataFrame
    """
    if categorical_features is None:
        categorical_features = detect_categorical_features(data)
    
    processed = data.copy()
    
    # Handle categorical features
    for col in categorical_features:
        if col in processed.columns:
            # Convert to category codes
            processed[col] = processed[col].astype('category').cat.codes
    
    # Handle numeric features
    numeric_features = [
        col for col in processed.columns 
        if col not in categorical_features
    ]
    
    for col in numeric_features:
        if col in processed.columns:
            # Standard scaling
            mean = processed[col].mean()
            std = processed[col].std()
            if std > 0:
                processed[col] = (processed[col] - mean) / std
    
    return processed
