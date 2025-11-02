"""
Main analyzer class for LENS
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings

from ..metrics import *
from ..utils.metrics_info import METRICS_INFO
from ..utils.normalization import normalize_scores
from ..visualization.plotter import plot_barplot, plot_heatmap


class LENSAnalyzer:
    """
    LENS Image Quality Analyzer
    
    Supports:
    - Single image analysis
    - Multiple image analysis
    - CSV batch processing
    - Normalized or raw scores
    - Scientific visualization
    """
    
    AVAILABLE_METRICS = list(METRICS_INFO.keys())
    
    def __init__(self, metrics: Optional[List[str]] = None, normalize: bool = False):
        """
        Initialize LENS Analyzer
        
        Args:
            metrics: List of metric names to compute. If None, uses all available metrics.
            normalize: Whether to normalize scores to 0-1 range (only for multiple images)
        """
        if metrics is None:
            self.metrics = self.AVAILABLE_METRICS
        else:
            # Validate metrics
            invalid = [m for m in metrics if m not in self.AVAILABLE_METRICS]
            if invalid:
                raise ValueError(f"Invalid metrics: {invalid}. Available: {self.AVAILABLE_METRICS}")
            self.metrics = metrics
        
        self.normalize = normalize
        self.results = None
        
    def compute_single_image(self, image_path: str) -> Dict[str, float]:
        """
        Compute metrics for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of metric_name: score
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        results = {}
        for metric_name in self.metrics:
            metric_class = self._get_metric_class(metric_name)
            try:
                score = metric_class().compute(img)
                results[metric_name] = score
            except Exception as e:
                warnings.warn(f"Failed to compute {metric_name}: {e}")
                results[metric_name] = None
        
        return results
    
    def compute_multiple_images(self, image_paths: List[str]) -> pd.DataFrame:
        """
        Compute metrics for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            DataFrame with columns: Image, Metric1, Metric2, ...
        """
        results = []
        
        for img_path in image_paths:
            row = {'Image': Path(img_path).name}
            scores = self.compute_single_image(img_path)
            row.update(scores)
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Normalize if requested
        if self.normalize and len(image_paths) > 1:
            df = self._normalize_dataframe(df)
        
        self.results = df
        return df
    
    def compute_from_csv(self, csv_path: str, image_column: str = 'Image') -> pd.DataFrame:
        """
        Compute metrics from CSV file containing image paths
        
        Args:
            csv_path: Path to CSV file
            image_column: Name of column containing image paths
            
        Returns:
            DataFrame with original columns + metric columns
        """
        # Try different CSV formats
        try:
            df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8')
        
        if image_column not in df.columns:
            raise ValueError(f"Column '{image_column}' not found in CSV")
        
        # Compute metrics for each row
        for metric_name in self.metrics:
            df[metric_name] = None
        
        for idx, row in df.iterrows():
            img_path = row[image_column]
            print(f"Processing {idx+1}/{len(df)}: {Path(img_path).name}")
            
            try:
                scores = self.compute_single_image(img_path)
                for metric_name, score in scores.items():
                    df.at[idx, metric_name] = score
            except Exception as e:
                warnings.warn(f"Failed to process {img_path}: {e}")
        
        # Normalize if requested
        if self.normalize and len(df) > 1:
            df = self._normalize_dataframe(df, preserve_columns=df.columns[:len(df.columns)-len(self.metrics)])
        
        self.results = df
        return df
    
    def plot_results(self, plot_type: str = 'bar', output_path: Optional[str] = None, **kwargs):
        """
        Visualize results
        
        Args:
            plot_type: 'bar' or 'heatmap'
            output_path: Path to save plot (optional)
            **kwargs: Additional arguments for plotting
        """
        if self.results is None:
            raise ValueError("No results to plot. Run compute_multiple_images() or compute_from_csv() first.")
        
        if plot_type == 'bar':
            plot_barplot(self.results, self.metrics, output_path=output_path, **kwargs)
        elif plot_type == 'heatmap':
            plot_heatmap(self.results, self.metrics, output_path=output_path, **kwargs)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Choose 'bar' or 'heatmap'")
    
    def save_results(self, output_path: str):
        """Save results to CSV"""
        if self.results is None:
            raise ValueError("No results to save")
        
        self.results.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
    
    def _get_metric_class(self, metric_name: str):
        """Get metric class from metric name"""
        from ..metrics import (
            BRISQUE, PIQE, LaplacianVariance, Tenengrad, 
            Brenner, EdgeDensity, Colorfulness, 
            RMSContrast, MichelsonContrast, NoiseEstimate
        )
        
        metric_map = {
            'BRISQUE': BRISQUE,
            'PIQE': PIQE,
            'Laplacian_Variance': LaplacianVariance,
            'Tenengrad': Tenengrad,
            'Brenner': Brenner,
            'Edge_Density': EdgeDensity,
            'Colorfulness': Colorfulness,
            'RMS_Contrast': RMSContrast,
            'Michelson_Contrast': MichelsonContrast,
            'Noise_Estimate': NoiseEstimate
        }
        
        return metric_map[metric_name]
    
    def _normalize_dataframe(self, df: pd.DataFrame, preserve_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize metric columns to 0-1 range"""
        df_norm = df.copy()
        
        for metric_name in self.metrics:
            if metric_name in df_norm.columns:
                values = df_norm[metric_name].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    
                    if max_val - min_val > 0:
                        normalized = (df_norm[metric_name] - min_val) / (max_val - min_val)
                        
                        # Invert if lower is better
                        if METRICS_INFO[metric_name]['better'] == 'lower':
                            normalized = 1 - normalized
                        
                        df_norm[metric_name + '_normalized'] = normalized
        
        return df_norm
    
    @staticmethod
    def list_metrics():
        """Print all available metrics"""
        from ..utils.metrics_info import print_metrics_info
        print_metrics_info()
    
    @staticmethod
    def get_metrics_table():
        """Get metrics information as DataFrame"""
        from ..utils.metrics_info import get_metrics_table
        return get_metrics_table()