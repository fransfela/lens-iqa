"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import cv2
import os
import tempfile


@pytest.fixture(scope='session')
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def create_test_image():
    """Factory fixture for creating test images"""
    def _create_image(height=100, width=100, channels=3, color=None):
        if color is None:
            img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            img = np.zeros((height, width, channels), dtype=np.uint8)
            img[:, :] = color
        return img
    
    return _create_image


@pytest.fixture
def create_gradient_image():
    """Create a gradient test image"""
    def _create_gradient(height=100, width=100):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            img[i, :, :] = int(i * 255 / height)
        return img
    
    return _create_gradient


@pytest.fixture
def save_test_image(test_data_dir):
    """Factory fixture for saving test images"""
    def _save_image(img, filename='test.png'):
        path = os.path.join(test_data_dir, filename)
        cv2.imwrite(path, img)
        return path
    
    return _save_image
```
