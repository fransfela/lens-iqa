"""
RMS Contrast - Root Mean Square Contrast
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class RMSContrast(BaseMetric):
    """
    RMS Contrast (Root Mean Square)
    
    Standard deviation of pixel intensities.
    Higher scores indicate more contrast.
    
    Range: 0-128 (for 8-bit images)
    Better: Higher
    
    Reference: Peli, 1990
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute RMS contrast
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            RMS contrast (higher is more contrast)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # RMS contrast = standard deviation
        contrast = float(gray.std())
        
        return contrast