"""
Brenner Focus Measure
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class Brenner(BaseMetric):
    """
    Brenner Focus Measure
    
    Fast focus measure based on horizontal gradient squared.
    Higher scores indicate better focus.
    
    Range: 0-∞ (scale-dependent)
    Better: Higher
    
    Reference: Brenner et al., 1976
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute Brenner focus measure
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Brenner score (higher is sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Horizontal gradient squared
        brenner = float(np.sum((gray[:, 2:] - gray[:, :-2])**2))
        
        return brenner