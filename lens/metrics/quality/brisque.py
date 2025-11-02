"""
BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
"""

import cv2
import numpy as np
from ...core.base_metric import BaseMetric

try:
    import torch
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False


class BRISQUE(BaseMetric):
    """
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    
    Measures naturalness based on scene statistics.
    Lower scores indicate better quality.
    
    Range: 0-100
    Better: Lower
    
    Reference: Mittal et al., 2012
    """
    
    def __init__(self):
        super().__init__()
        if not HAS_PIQ:
            raise ImportError(
                "BRISQUE requires 'piq' library. "
                "Install with: pip install piq torch"
            )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute(self, image: np.ndarray) -> float:
        """
        Compute BRISQUE score
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            BRISQUE score (0-100, lower is better)
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Compute BRISQUE
        with torch.no_grad():
            score = piq.brisque(img_tensor, data_range=1.0, reduction='none')
        
        return float(score.item())