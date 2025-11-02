"""
Noise Estimate using high-pass filter
"""

import cv2
import numpy as np
from scipy import ndimage
from ...core.base_metric import BaseMetric


class NoiseEstimate(BaseMetric):
    """
    Noise Estimate (High-pass Filter)
    
    Estimates noise level using Laplacian high-pass filter.
    Lower scores indicate less noise.
    
    Range: 0-50 (typical)
    Better: Lower
    
    Reference: Immerkaer, 1996
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute noise estimate
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Noise estimate (lower is cleaner)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        H, W = gray.shape
        
        # High-pass filter kernel
        M = np.array([[1, -2, 1],
                      [-2, 4, -2],
                      [1, -2, 1]])
        
        # Apply filter
        sigma = np.sum(np.sum(np.absolute(ndimage.convolve(gray, M))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
        
        return float(sigma)