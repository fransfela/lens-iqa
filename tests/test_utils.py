"""
Unit tests for utility functions
"""

import pytest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import tempfile
import os

from lens.utils.image_io import (
    load_image, save_image, load_images_from_paths,
    validate_image, get_image_info, resize_image,
    convert_color_space
)
from lens.utils.normalization import (
    normalize_metric, normalize_scores, normalize_dataframe,
    denormalize_metric, scale_to_range
)


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image():
    """Create sample image"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Blue channel
    return img


class TestImageIO:
    """Tests for image I/O functions"""
    
    def test_load_image(self, temp_dir, sample_image):
        # Save image
        img_path = os.path.join(temp_dir, 'test.png')
        cv2.imwrite(img_path, sample_image)
        
        # Load image
        loaded = load_image(img_path)
        
        assert loaded is not None
        assert loaded.shape == sample_image.shape
        assert np.array_equal(loaded, sample_image)
    
    def test_load_nonexistent_image(self):
        loaded = load_image('nonexistent.png')
        assert loaded is None
    
    def test_save_image(self, temp_dir, sample_image):
        output_path = os.path.join(temp_dir, 'output.png')
        success = save_image(sample_image, output_path)
        
        assert success == True
        assert os.path.exists(output_path)
    
    def test_load_multiple_images(self, temp_dir):
        # Create multiple images
        paths = []
        for i in range(3):
            img = np.ones((50, 50, 3), dtype=np.uint8) * (i * 100)
            path = os.path.join(temp_dir, f'img_{i}.png')
            cv2.imwrite(path, img)
            paths.append(path)
        
        # Load all
        images = load_images_from_paths(paths, verbose=False)
        
        assert len(images) == 3
        assert all(img is not None for img in images)
    
    def test_validate_image(self, sample_image):
        assert validate_image(sample_image) == True
        assert validate_image(None) == False
        assert validate_image(np.array([1, 2, 3])) == False
    
    def test_get_image_info(self, sample_image):
        info = get_image_info(sample_image)
        
        assert info['height'] == 100
        assert info['width'] == 100
        assert info['channels'] == 3
        assert info['dtype'] == 'uint8'
    
    def test_resize_image(self, sample_image):
        resized = resize_image(sample_image, width=50, height=50)
        
        assert resized.shape == (50, 50, 3)
    
    def test_resize_with_scale(self, sample_image):
        resized = resize_image(sample_image, scale=0.5)
        
        assert resized.shape == (50, 50, 3)
    
    def test_convert_color_space(self, sample_image):
        rgb = convert_color_space(sample_image, 'BGR2RGB')
        
        assert rgb.shape == sample_image.shape
        assert not np.array_equal(rgb, sample_image)  # Should be different


class TestNormalization:
    """Tests for normalization functions"""
    
    def test_normalize_metric_basic(self):
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_metric(values, 'Laplacian_Variance')
        
        assert len(normalized) == len(values)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_normalize_lower_is_better(self):
        values = np.array([10.0, 20.0, 30.0])
        normalized = normalize_metric(values, 'PIQE')  # Lower is better
        
        # After normalization and inversion:
        # 10 (lowest) should become highest normalized value
        assert normalized[0] > normalized[2]
    
    def test_normalize_with_nan(self):
        values = np.array([10.0, np.nan, 30.0])
        normalized = normalize_metric(values, 'Laplacian_Variance')
        
        assert not np.isnan(normalized[0])
        assert np.isnan(normalized[1])
        assert not np.isnan(normalized[2])
    
    def test_normalize_constant_values(self):
        values = np.array([10.0, 10.0, 10.0])
        normalized = normalize_metric(values, 'Laplacian_Variance')
        
        # All values should be 0.5 when constant
        assert np.allclose(normalized, 0.5)
    
    def test_normalize_scores_dict(self):
        scores = {
            'PIQE': [10, 20, 30],
            'Laplacian_Variance': [100, 200, 300]
        }
        
        normalized = normalize_scores(scores)
        
        assert 'PIQE' in normalized
        assert 'Laplacian_Variance' in normalized
        assert len(normalized['PIQE']) == 3
    
    def test_normalize_scores_scalar(self):
        scores = {
            'PIQE': 25.0,
            'Laplacian_Variance': 150.0
        }
        
        normalized = normalize_scores(scores)
        
        assert isinstance(normalized['PIQE'], float)
        assert isinstance(normalized['Laplacian_Variance'], float)
    
    def test_normalize_dataframe(self):
        df = pd.DataFrame({
            'Image': ['img1', 'img2', 'img3'],
            'PIQE': [10, 20, 30],
            'Laplacian_Variance': [100, 200, 300]
        })
        
        df_norm = normalize_dataframe(df, ['PIQE', 'Laplacian_Variance'])
        
        assert 'PIQE_normalized' in df_norm.columns
        assert 'Laplacian_Variance_normalized' in df_norm.columns
        assert len(df_norm) == len(df)
    
    def test_denormalize_metric(self):
        normalized = np.array([0.0, 0.5, 1.0])
        denormalized = denormalize_metric(normalized, 'Laplacian_Variance', 0, 100)
        
        assert np.allclose(denormalized, [0, 50, 100])
    
    def test_scale_to_range(self):
        values = np.array([0, 50, 100])
        scaled = scale_to_range(values, 0, 1)
        
        assert np.allclose(scaled, [0, 0.5, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])