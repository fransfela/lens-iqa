"""
Edge Density - Detail richness metric
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class EdgeDensity(BaseMetric):
    """
    Edge Density
    
    Proportion of edge pixels detected using Canny edge detector.
    Higher scores indicate more edges/details.
    
    Range: 0-1
    Better: Higher
    
    Reference: Canny, 1986
    """
    
    def __init__(self, threshold1: int = 100, threshold2: int = 200):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute edge density
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            Edge density (0-1, higher is more detailed)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        
        # Edge density
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        return edge_density