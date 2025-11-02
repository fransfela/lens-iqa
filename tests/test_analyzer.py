"""
Unit tests for LENSAnalyzer
"""

import pytest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import tempfile
import os

from lens import LENSAnalyzer
from lens.utils.metrics_info import METRICS_INFO


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_images(temp_dir):
    """Create sample test images"""
    images = []
    
    for i in range(3):
        # Create different images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, i % 3] = (i + 1) * 80
        
        # Save image
        img_path = os.path.join(temp_dir, f'test_image_{i}.png')
        cv2.imwrite(img_path, img)
        images.append(img_path)
    
    return images


@pytest.fixture
def sample_csv(temp_dir, sample_images):
    """Create sample CSV file"""
    data = {
        'Image': sample_images,
        'Method': ['Original', 'Enhanced', 'Processed'],
        'Quality': ['Low', 'High', 'Medium']
    }
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path


class TestLENSAnalyzerInitialization:
    """Tests for LENSAnalyzer initialization"""
    
    def test_default_initialization(self):
        analyzer = LENSAnalyzer()
        
        assert analyzer.normalize == False
        assert len(analyzer.metrics) > 0
        assert analyzer.results is None
    
    def test_custom_metrics(self):
        metrics = ['PIQE', 'Laplacian_Variance']
        analyzer = LENSAnalyzer(metrics=metrics)
        
        assert analyzer.metrics == metrics
    
    def test_normalization_flag(self):
        analyzer = LENSAnalyzer(normalize=True)
        
        assert analyzer.normalize == True
    
    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            LENSAnalyzer(metrics=['InvalidMetric'])


class TestSingleImageAnalysis:
    """Tests for single image analysis"""
    
    def test_compute_single_image(self, sample_images):
        analyzer = LENSAnalyzer()
        scores = analyzer.compute_single_image(sample_images[0])
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        
        # Check that scores are numbers
        for metric, score in scores.items():
            if score is not None:
                assert isinstance(score, (int, float))
    
    def test_compute_selected_metrics(self, sample_images):
        analyzer = LENSAnalyzer(metrics=['PIQE', 'Laplacian_Variance'])
        scores = analyzer.compute_single_image(sample_images[0])
        
        assert 'PIQE' in scores
        assert 'Laplacian_Variance' in scores
        assert len(scores) == 2
    
    def test_invalid_image_path(self):
        analyzer = LENSAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.compute_single_image('nonexistent_image.png')


class TestMultipleImagesAnalysis:
    """Tests for multiple images analysis"""
    
    def test_compute_multiple_images(self, sample_images):
        analyzer = LENSAnalyzer(normalize=False)
        results = analyzer.compute_multiple_images(sample_images)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_images)
        assert 'Image' in results.columns
    
    def test_compute_with_normalization(self, sample_images):
        analyzer = LENSAnalyzer(normalize=True)
        results = analyzer.compute_multiple_images(sample_images)
        
        # Check for normalized columns
        normalized_cols = [col for col in results.columns if '_normalized' in col]
        assert len(normalized_cols) > 0
        
        # Check normalized values are in 0-1 range
        for col in normalized_cols:
            values = results[col].dropna()
            assert values.min() >= 0
            assert values.max() <= 1
    
    def test_results_stored(self, sample_images):
        analyzer = LENSAnalyzer()
        results = analyzer.compute_multiple_images(sample_images)
        
        assert analyzer.results is not None
        assert len(analyzer.results) == len(sample_images)


class TestCSVBatchProcessing:
    """Tests for CSV batch processing"""
    
    def test_compute_from_csv(self, sample_csv):
        analyzer = LENSAnalyzer()
        results = analyzer.compute_from_csv(sample_csv, image_column='Image')
        
        assert isinstance(results, pd.DataFrame)
        assert 'Image' in results.columns
        assert 'Method' in results.columns
        
        # Check that metric columns were added
        metric_cols = [col for col in results.columns 
                      if col in analyzer.metrics]
        assert len(metric_cols) > 0
    
    def test_csv_with_normalization(self, sample_csv):
        analyzer = LENSAnalyzer(normalize=True)
        results = analyzer.compute_from_csv(sample_csv, image_column='Image')
        
        # Check for normalized columns
        normalized_cols = [col for col in results.columns if '_normalized' in col]
        assert len(normalized_cols) > 0
    
    def test_invalid_column_name(self, sample_csv):
        analyzer = LENSAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.compute_from_csv(sample_csv, image_column='NonexistentColumn')


class TestVisualization:
    """Tests for visualization functions"""
    
    def test_plot_barplot(self, sample_images, temp_dir):
        analyzer = LENSAnalyzer()
        analyzer.compute_multiple_images(sample_images)
        
        output_path = os.path.join(temp_dir, 'barplot.png')
        analyzer.plot_results(plot_type='bar', output_path=output_path)
        
        assert os.path.exists(output_path)
    
    def test_plot_heatmap(self, sample_images, temp_dir):
        analyzer = LENSAnalyzer()
        analyzer.compute_multiple_images(sample_images)
        
        output_path = os.path.join(temp_dir, 'heatmap.png')
        analyzer.plot_results(plot_type='heatmap', output_path=output_path)
        
        assert os.path.exists(output_path)
    
    def test_plot_without_results(self):
        analyzer = LENSAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.plot_results(plot_type='bar')
    
    def test_invalid_plot_type(self, sample_images):
        analyzer = LENSAnalyzer()
        analyzer.compute_multiple_images(sample_images)
        
        with pytest.raises(ValueError):
            analyzer.plot_results(plot_type='invalid')


class TestSaveResults:
    """Tests for saving results"""
    
    def test_save_results(self, sample_images, temp_dir):
        analyzer = LENSAnalyzer()
        analyzer.compute_multiple_images(sample_images)
        
        output_path = os.path.join(temp_dir, 'results.csv')
        analyzer.save_results(output_path)
        
        assert os.path.exists(output_path)
        
        # Load and verify
        df = pd.read_csv(output_path)
        assert len(df) == len(sample_images)
    
    def test_save_without_results(self, temp_dir):
        analyzer = LENSAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.save_results(os.path.join(temp_dir, 'results.csv'))


class TestStaticMethods:
    """Tests for static methods"""
    
    def test_list_metrics(self, capsys):
        LENSAnalyzer.list_metrics()
        
        captured = capsys.readouterr()
        assert 'LENS' in captured.out
        assert 'BRISQUE' in captured.out or 'PIQE' in captured.out
    
    def test_get_metrics_table(self):
        df = LENSAnalyzer.get_metrics_table()
        
        assert isinstance(df, pd.DataFrame)
        assert 'Metric' in df.columns
        assert len(df) > 0


class TestNormalization:
    """Tests for normalization behavior"""
    
    def test_normalization_inverts_lower_is_better(self, sample_images):
        analyzer = LENSAnalyzer(metrics=['PIQE'], normalize=True)
        results = analyzer.compute_multiple_images(sample_images)
        
        # PIQE: lower is better, so after normalization:
        # - lowest PIQE should become highest normalized value
        # - highest PIQE should become lowest normalized value
        
        if 'PIQE_normalized' in results.columns:
            raw = results['PIQE'].values
            norm = results['PIQE_normalized'].values
            
            # Check that order is reversed
            raw_order = np.argsort(raw)
            norm_order = np.argsort(norm)[::-1]
            
            # Orders should match (inverted)
            assert np.allclose(raw_order, norm_order) or len(raw) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])