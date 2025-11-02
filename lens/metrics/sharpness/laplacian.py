"""
Laplacian Variance - Sharpness metric
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class LaplacianVariance(BaseMetric):
    """
    Laplacian Variance
    
    Measures edge sharpness using second derivative (Laplacian operator).
    Higher scores indicate sharper images.
    
    Range: 0-5000+ (typical)
    Better: Higher
    
    Reference: Pech-Pacheco et al., 2000
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute Laplacian variance
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Laplacian variance (higher is sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Variance
        variance = float(laplacian.var())
        
        return variance