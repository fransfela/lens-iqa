"""
Colorfulness - Hasler & Süsstrunk metric
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class Colorfulness(BaseMetric):
    """
    Colorfulness (Hasler & Süsstrunk)
    
    Perceptual colorfulness metric based on opponent color space.
    Higher scores indicate more colorful images.
    
    Range: 0-150 (typical)
    Better: Higher
    
    Reference: Hasler & Süsstrunk, 2003
    """
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute colorfulness score
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Colorfulness score (higher is more colorful)
        """
        # Split channels (BGR)
        B, G, R = cv2.split(image.astype(np.float32))
        
        # Compute rg = R - G
        rg = np.absolute(R - G)
        
        # Compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        
        # Compute statistics
        rg_mean, rg_std = np.mean(rg), np.std(rg)
        yb_mean, yb_std = np.mean(yb), np.std(yb)
        
        # Combine
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
        
        colorfulness = float(std_root + (0.3 * mean_root))
        
        return colorfulness