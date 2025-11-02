"""
Score normalization utilities for LENS
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from .metrics_info import METRICS_INFO


def normalize_metric(values: np.ndarray, 
                    metric_name: str,
                    method: str = 'min-max') -> np.ndarray:
    """
    Normalize metric values to 0-1 range
    
    Args:
        values: Array of metric values
        metric_name: Name of the metric
        method: Normalization method ('min-max', 'z-score')
        
    Returns:
        Normalized values (0-1 range)
        
    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> normalized = normalize_metric(values, 'BRISQUE')
        >>> print(normalized)
        [1.0, 0.75, 0.5, 0.25, 0.0]  # Lower is better, so inverted
    """
    # Remove NaN values for computation
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return values
    
    # Min-max normalization
    if method == 'min-max':
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val - min_val == 0:
            normalized = np.zeros_like(values)
            normalized[valid_mask] = 0.5
        else:
            normalized = np.zeros_like(values)
            normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)
    
    # Z-score normalization (then clip to 0-1)
    elif method == 'z-score':
        mean_val = valid_values.mean()
        std_val = valid_values.std()
        
        if std_val == 0:
            normalized = np.zeros_like(values)
            normalized[valid_mask] = 0.5
        else:
            normalized = np.zeros_like(values)
            z_scores = (valid_values - mean_val) / std_val
            # Map to 0-1 using sigmoid
            normalized[valid_mask] = 1 / (1 + np.exp(-z_scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Invert if lower is better
    metric_info = METRICS_INFO.get(metric_name, {})
    if metric_info.get('better', 'higher') == 'lower':
        normalized[valid_mask] = 1 - normalized[valid_mask]
    
    return normalized


def normalize_scores(scores: Dict[str, Union[float, List[float]]], 
                    metric_names: Optional[List[str]] = None) -> Dict[str, Union[float, List[float]]]:
    """
    Normalize multiple metric scores
    
    Args:
        scores: Dictionary of metric_name: value(s)
        metric_names: List of metrics to normalize (None = all)
        
    Returns:
        Dictionary with normalized scores
        
    Examples:
        >>> scores = {
        ...     'BRISQUE': [10, 20, 30],
        ...     'Laplacian_Variance': [100, 200, 300]
        ... }
        >>> normalized = normalize_scores(scores)
    """
    if metric_names is None:
        metric_names = list(scores.keys())
    
    normalized_scores = {}
    
    for metric_name in metric_names:
        if metric_name not in scores:
            continue
        
        values = scores[metric_name]
        
        # Convert to array if necessary
        if isinstance(values, (int, float)):
            values = np.array([values])
            is_scalar = True
        else:
            values = np.array(values)
            is_scalar = False
        
        # Normalize
        normalized = normalize_metric(values, metric_name)
        
        # Convert back to scalar if input was scalar
        if is_scalar:
            normalized_scores[metric_name] = float(normalized[0])
        else:
            normalized_scores[metric_name] = normalized.tolist()
    
    return normalized_scores


def normalize_dataframe(df: pd.DataFrame, 
                       metric_columns: List[str],
                       method: str = 'min-max',
                       suffix: str = '_normalized') -> pd.DataFrame:
    """
    Normalize metric columns in a DataFrame
    
    Args:
        df: Input DataFrame
        metric_columns: List of column names to normalize
        method: Normalization method
        suffix: Suffix for normalized column names
        
    Returns:
        DataFrame with additional normalized columns
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'Image': ['img1', 'img2', 'img3'],
        ...     'BRISQUE': [10, 20, 30],
        ...     'Laplacian_Variance': [100, 200, 300]
        ... })
        >>> df_norm = normalize_dataframe(df, ['BRISQUE', 'Laplacian_Variance'])
    """
    df_normalized = df.copy()
    
    for metric in metric_columns:
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        normalized = normalize_metric(values, metric, method=method)
        
        df_normalized[metric + suffix] = normalized
    
    return df_normalized


def get_normalized_range(metric_name: str) -> tuple:
    """
    Get the expected normalized range for a metric
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Tuple of (min, max) normalized values
        
    Examples:
        >>> get_normalized_range('BRISQUE')
        (0.0, 1.0)
    """
    return (0.0, 1.0)


def denormalize_metric(normalized_values: np.ndarray,
                      metric_name: str,
                      original_min: float,
                      original_max: float) -> np.ndarray:
    """
    Convert normalized values back to original scale
    
    Args:
        normalized_values: Normalized values (0-1)
        metric_name: Name of the metric
        original_min: Original minimum value
        original_max: Original maximum value
        
    Returns:
        Denormalized values in original scale
        
    Examples:
        >>> normalized = np.array([0.0, 0.5, 1.0])
        >>> original = denormalize_metric(normalized, 'BRISQUE', 0, 100)
        >>> print(original)
        [100, 50, 0]  # Inverted because lower is better
    """
    # Invert if lower is better
    metric_info = METRICS_INFO.get(metric_name, {})
    if metric_info.get('better', 'higher') == 'lower':
        normalized_values = 1 - normalized_values
    
    # Denormalize
    denormalized = normalized_values * (original_max - original_min) + original_min
    
    return denormalized


def scale_to_range(values: np.ndarray, 
                  target_min: float = 0.0, 
                  target_max: float = 1.0) -> np.ndarray:
    """
    Scale values to a target range
    
    Args:
        values: Input values
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        Scaled values
        
    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> scaled = scale_to_range(values, 0, 100)
    """
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return values
    
    min_val = valid_values.min()
    max_val = valid_values.max()
    
    if max_val - min_val == 0:
        scaled = np.full_like(values, (target_min + target_max) / 2)
    else:
        scaled = np.zeros_like(values)
        normalized = (valid_values - min_val) / (max_val - min_val)
        scaled[valid_mask] = normalized * (target_max - target_min) + target_min
    
    return scaled