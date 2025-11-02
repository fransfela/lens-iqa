"""
Tenengrad - Focus quality metric
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class Tenengrad(BaseMetric):
    """
    Tenengrad Gradient
    
    Focus measure based on gradient magnitude (Sobel operator).
    Higher scores indicate better focus.
    
    Range: 0-∞ (scale-dependent)
    Better: Higher
    
    Reference: Krotkov, 1988
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute Tenengrad score
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Tenengrad score (higher is sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Tenengrad: sum of squared gradients above threshold
        threshold = np.mean(gradient_magnitude)
        tenengrad = float(np.sum(gradient_magnitude[gradient_magnitude > threshold]**2))
        
        return tenengrad