"""
Michelson Contrast
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class MichelsonContrast(BaseMetric):
    """
    Michelson Contrast
    
    Contrast ratio: (L_max - L_min) / (L_max + L_min)
    Higher scores indicate more contrast.
    
    Range: 0-1
    Better: Higher
    
    Reference: Michelson, 1927
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute Michelson contrast
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Michelson contrast (0-1, higher is more contrast)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        L_max = gray.max()
        L_min = gray.min()
        
        if L_max + L_min == 0:
            return 0.0
        
        contrast = float((L_max - L_min) / (L_max + L_min))
        
        return contrast