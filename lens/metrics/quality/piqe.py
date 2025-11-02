"""
PIQE: Perception-based Image Quality Evaluator
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric


class PIQE(BaseMetric):
    """
    PIQE (Perception-based Image Quality Evaluator)
    
    Block-based distortion estimation for perceptual quality.
    Lower scores indicate better quality.
    
    Range: 0-100
    Better: Lower
    
    Reference: Venkatanath et al., 2015
    """
    
    def __init__(self, block_size: int = 16):
        super().__init__()
        self.block_size = block_size
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute PIQE score
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            PIQE score (0-100, lower is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        h_blocks = h // self.block_size
        w_blocks = w // self.block_size
        
        distortions = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*self.block_size:(i+1)*self.block_size, 
                            j*self.block_size:(j+1)*self.block_size]
                
                # Compute local statistics
                mu = np.mean(block)
                sigma = np.std(block)
                
                # Distortion measure
                distortion = sigma / (mu + 1e-6)
                distortions.append(distortion)
        
        # PIQE score
        piqe_score = np.mean(distortions) * 100
        
        return float(piqe_score)