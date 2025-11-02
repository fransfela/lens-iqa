"""
Unit tests for individual metrics
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from lens.metrics import (
    BRISQUE, PIQE, LaplacianVariance, Tenengrad,
    Brenner, EdgeDensity, Colorfulness, RMSContrast,
    MichelsonContrast, NoiseEstimate
)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a 100x100 color image with gradient
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[i, :, :] = i * 255 // 100
    return img


@pytest.fixture
def gray_image():
    """Create a sample grayscale image"""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


@pytest.fixture
def noisy_image():
    """Create a noisy image"""
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return img


class TestPIQE:
    """Tests for PIQE metric"""
    
    def test_piqe_initialization(self):
        metric = PIQE()
        assert metric.name == 'PIQE'
        assert metric.block_size == 16
    
    def test_piqe_compute(self, sample_image):
        metric = PIQE()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0
        assert score <= 200  # Reasonable upper bound
    
    def test_piqe_custom_block_size(self, sample_image):
        metric = PIQE(block_size=32)
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0


class TestLaplacianVariance:
    """Tests for Laplacian Variance metric"""
    
    def test_laplacian_initialization(self):
        metric = LaplacianVariance()
        assert metric.name == 'LaplacianVariance'
    
    def test_laplacian_compute(self, sample_image):
        metric = LaplacianVariance()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_laplacian_sharp_vs_blurry(self, sample_image):
        metric = LaplacianVariance()
        
        # Sharp image
        sharp_score = metric.compute(sample_image)
        
        # Blurry image
        blurry = cv2.GaussianBlur(sample_image, (15, 15), 0)
        blurry_score = metric.compute(blurry)
        
        # Sharp should have higher score
        assert sharp_score > blurry_score


class TestTenengrad:
    """Tests for Tenengrad metric"""
    
    def test_tenengrad_initialization(self):
        metric = Tenengrad()
        assert metric.name == 'Tenengrad'
    
    def test_tenengrad_compute(self, sample_image):
        metric = Tenengrad()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0


class TestBrenner:
    """Tests for Brenner metric"""
    
    def test_brenner_initialization(self):
        metric = Brenner()
        assert metric.name == 'Brenner'
    
    def test_brenner_compute(self, sample_image):
        metric = Brenner()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0


class TestEdgeDensity:
    """Tests for Edge Density metric"""
    
    def test_edge_density_initialization(self):
        metric = EdgeDensity()
        assert metric.name == 'EdgeDensity'
    
    def test_edge_density_compute(self, sample_image):
        metric = EdgeDensity()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_edge_density_range(self, gray_image):
        metric = EdgeDensity()
        score = metric.compute(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR))
        
        assert 0 <= score <= 1


class TestColorfulness:
    """Tests for Colorfulness metric"""
    
    def test_colorfulness_initialization(self):
        metric = Colorfulness()
        assert metric.name == 'Colorfulness'
    
    def test_colorfulness_compute(self, sample_image):
        metric = Colorfulness()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_colorfulness_gray_vs_color(self):
        metric = Colorfulness()
        
        # Grayscale image (low colorfulness)
        gray = np.ones((100, 100, 3), dtype=np.uint8) * 128
        gray_score = metric.compute(gray)
        
        # Colorful image
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[:, :, 0] = 255  # Red
        color[:, :, 1] = 0    # Green
        color[:, :, 2] = 255  # Blue
        color_score = metric.compute(color)
        
        # Colorful should have higher score
        assert color_score > gray_score


class TestRMSContrast:
    """Tests for RMS Contrast metric"""
    
    def test_rms_contrast_initialization(self):
        metric = RMSContrast()
        assert metric.name == 'RMSContrast'
    
    def test_rms_contrast_compute(self, sample_image):
        metric = RMSContrast()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert 0 <= score <= 128


class TestMichelsonContrast:
    """Tests for Michelson Contrast metric"""
    
    def test_michelson_initialization(self):
        metric = MichelsonContrast()
        assert metric.name == 'MichelsonContrast'
    
    def test_michelson_compute(self, sample_image):
        metric = MichelsonContrast()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_michelson_high_contrast(self):
        metric = MichelsonContrast()
        
        # High contrast image (black and white)
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[50:, :] = 255
        score = metric.compute(high_contrast)
        
        assert score > 0.9  # Should be close to 1


class TestNoiseEstimate:
    """Tests for Noise Estimate metric"""
    
    def test_noise_initialization(self):
        metric = NoiseEstimate()
        assert metric.name == 'NoiseEstimate'
    
    def test_noise_compute(self, sample_image):
        metric = NoiseEstimate()
        score = metric.compute(sample_image)
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_noise_clean_vs_noisy(self, sample_image, noisy_image):
        metric = NoiseEstimate()
        
        clean_score = metric.compute(sample_image)
        noisy_score = metric.compute(noisy_image)
        
        # Noisy image should have higher score
        assert noisy_score > clean_score


class TestMetricCallable:
    """Test that metrics can be called as functions"""
    
    def test_callable(self, sample_image):
        metric = LaplacianVariance()
        
        # Test both methods
        score1 = metric.compute(sample_image)
        score2 = metric(sample_image)
        
        assert score1 == score2


class TestMetricRepr:
    """Test metric string representation"""
    
    def test_repr(self):
        metric = LaplacianVariance()
        assert repr(metric) == "LaplacianVariance()"


# Skip BRISQUE tests if PyTorch not available
try:
    import torch
    import piq
    
    class TestBRISQUE:
        """Tests for BRISQUE metric"""
        
        def test_brisque_initialization(self):
            metric = BRISQUE()
            assert metric.name == 'BRISQUE'
        
        def test_brisque_compute(self, sample_image):
            metric = BRISQUE()
            score = metric.compute(sample_image)
            
            assert isinstance(score, float)
            assert 0 <= score <= 100

except ImportError:
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])