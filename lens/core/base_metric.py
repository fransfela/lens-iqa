"""
Base class for all image quality metrics
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """Abstract base class for all metrics"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute(self, image: np.ndarray) -> float:
        """
        Compute metric for given image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Metric score as float
        """
        pass
    
    def __call__(self, image: np.ndarray) -> float:
        """Allow calling metric as function"""
        return self.compute(image)
    
    def __repr__(self):
        return f"{self.name}()"