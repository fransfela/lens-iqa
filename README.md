# LENS 🔍

**L**ibrary for **E**valuating **N**o-reference image quality **S**cores

A comprehensive Python toolkit for assessing image quality using 10 no-reference metrics, with support for batch processing and scientific visualization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/lens-iqa.svg)](https://badge.fury.io/py/lens-iqa)

---

## ✨ Features

- 🎯 **10 No-Reference Metrics** - BRISQUE, PIQE, Laplacian Variance, Tenengrad, Brenner, Edge Density, Colorfulness, RMS Contrast, Michelson Contrast, Noise Estimate
- 📊 **Flexible Input** - Single image, multiple images, or CSV batch processing
- 📈 **Auto-Normalization** - Optional score normalization to 0-1 range for fair comparison
- 🎨 **Publication-Ready Plots** - Professional barplots and heatmaps
- 🚀 **Fast & Lightweight** - Minimal dependencies (OpenCV, NumPy, SciPy)
- 🔧 **Extensible** - Easy to add custom metrics
- 📖 **Well-Documented** - Comprehensive examples and API reference

---

## 📊 Supported Metrics

| Metric | Full Name | Category | Range | Better | Description |
|--------|-----------|----------|-------|--------|-------------|
| **BRISQUE** | Blind/Referenceless Image Spatial Quality Evaluator | Quality | 0-100 | Lower | Measures naturalness based on scene statistics |
| **PIQE** | Perception-based Image Quality Evaluator | Quality | 0-100 | Lower | Block-based distortion estimation |
| **Laplacian_Variance** | Laplacian Variance | Sharpness | 0-5000+ | Higher | Edge sharpness using second derivative |
| **Tenengrad** | Tenengrad Gradient | Sharpness | 0-∞ | Higher | Focus measure based on gradient magnitude |
| **Brenner** | Brenner Focus Measure | Sharpness | 0-∞ | Higher | Fast focus measure using horizontal gradient |
| **Edge_Density** | Edge Density | Sharpness | 0-1 | Higher | Proportion of edge pixels (Canny detector) |
| **Colorfulness** | Colorfulness (Hasler & Süsstrunk) | Color | 0-150 | Higher | Perceptual colorfulness in opponent color space |
| **RMS_Contrast** | RMS Contrast | Color | 0-128 | Higher | Standard deviation of pixel intensities |
| **Michelson_Contrast** | Michelson Contrast | Color | 0-1 | Higher | Contrast ratio (L_max - L_min)/(L_max + L_min) |
| **Noise_Estimate** | Noise Estimate | Noise | 0-50 | Lower | Noise level using high-pass filter |

### 📚 References

- **BRISQUE**: Mittal et al., "No-Reference Image Quality Assessment in the Spatial Domain", IEEE TIP 2012
- **PIQE**: Venkatanath et al., "Blind Image Quality Evaluation Using Perception Based Features", NCC 2015
- **Laplacian Variance**: Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy", ICPR 2000
- **Colorfulness**: Hasler & Süsstrunk, "Measuring colorfulness in natural images", SPIE 2003

---

## 📦 Installation

### Quick Install (Basic Metrics)
```bash
pip install lens-iqa
```

### Full Install (Including BRISQUE)
```bash
pip install lens-iqa[full]
```

This installs PyTorch and PIQ for BRISQUE metric support.

### Install from Source
```bash
git clone https://github.com/yourusername/lens.git
cd lens
pip install -e .
```

### Requirements

**Core dependencies:**
- Python >= 3.8
- NumPy >= 1.20.0
- OpenCV >= 4.5.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

**Optional (for BRISQUE):**
- PyTorch >= 1.9.0
- PIQ >= 0.7.0

---

## 🚀 Quick Start

### 1. Single Image Analysis
```python
from lens import LENSAnalyzer

# Compute all metrics for a single image
analyzer = LENSAnalyzer()
scores = analyzer.compute_single_image('image.png')

print(scores)
# Output:
# {
#     'BRISQUE': 25.3,
#     'PIQE': 18.7,
#     'Laplacian_Variance': 1234.5,
#     'Tenengrad': 5678901.2,
#     ...
# }
```

### 2. Multiple Images with Visualization
```python
from lens import LENSAnalyzer

# List of images to analyze
images = ['img1.png', 'img2.png', 'img3.png']

# Analyze with normalized scores
analyzer = LENSAnalyzer(normalize=True)
results = analyzer.compute_multiple_images(images)

# View results
print(results)

# Generate barplot
analyzer.plot_results(plot_type='bar', output_path='barplot.png')

# Generate heatmap
analyzer.plot_results(plot_type='heatmap', output_path='heatmap.png')

# Save results to CSV
analyzer.save_results('results.csv')
```

### 3. CSV Batch Processing
```python
from lens import LENSAnalyzer

# CSV file with image paths
# Example CSV:
# Image,Method,Pose
# /path/to/img1.png,bicubic,front
# /path/to/img2.png,SR_model,front

analyzer = LENSAnalyzer(
    metrics=['BRISQUE', 'Laplacian_Variance', 'Colorfulness'],
    normalize=True
)

# Compute metrics for all images in CSV
results = analyzer.compute_from_csv(
    'dataset.csv', 
    image_column='Image'  # Column name with image paths
)

# Save results (adds metric columns to original CSV)
analyzer.save_results('dataset_with_metrics.csv')

# Visualize
analyzer.plot_results(plot_type='heatmap', output_path='results.png')
```

### 4. Select Specific Metrics
```python
from lens import LENSAnalyzer

# Only compute sharpness and quality metrics
analyzer = LENSAnalyzer(
    metrics=['BRISQUE', 'PIQE', 'Laplacian_Variance', 'Tenengrad']
)

scores = analyzer.compute_single_image('image.png')
```

### 5. Use Individual Metrics
```python
from lens.metrics import LaplacianVariance, BRISQUE, Colorfulness
import cv2

# Load image
img = cv2.imread('image.png')

# Compute individual metrics
sharpness = LaplacianVariance().compute(img)
quality = BRISQUE().compute(img)
color = Colorfulness().compute(img)

print(f"Sharpness: {sharpness:.2f}")
print(f"Quality (BRISQUE): {quality:.2f}")
print(f"Colorfulness: {color:.2f}")
```

---

## 📖 Documentation

### View Available Metrics
```python
from lens import LENSAnalyzer

# Print detailed information about all metrics
LENSAnalyzer.list_metrics()

# Get metrics as DataFrame
df = LENSAnalyzer.get_metrics_table()
print(df)
```

### Normalization

When analyzing **multiple images**, you can normalize scores to 0-1 range for fair comparison:
```python
analyzer = LENSAnalyzer(normalize=True)
results = analyzer.compute_multiple_images(['img1.png', 'img2.png'])

# Results will have both raw and normalized scores:
# - Original metric columns (e.g., 'BRISQUE')
# - Normalized columns (e.g., 'BRISQUE_normalized')
```

**Normalization rules:**
- Metrics where "lower is better" (BRISQUE, PIQE, Noise_Estimate) are inverted: `1 - normalized_value`
- Metrics where "higher is better" remain as-is
- After normalization, **higher always means better quality**

---

## 📊 Example Outputs

### Barplot Example
![Barplot](docs/images/barplot_example.png)

### Heatmap Example
![Heatmap](docs/images/heatmap_example.png)

---

## 🔧 Advanced Usage

### Custom Metric Parameters
```python
from lens.metrics import EdgeDensity, PIQE

# Edge density with custom Canny thresholds
edge_metric = EdgeDensity(threshold1=50, threshold2=150)
score = edge_metric.compute(img)

# PIQE with custom block size
piqe_metric = PIQE(block_size=32)
score = piqe_metric.compute(img)
```

### Extend with Custom Metrics
```python
from lens.core.base_metric import BaseMetric
import cv2
import numpy as np

class MyCustomMetric(BaseMetric):
    """My custom quality metric"""
    
    def compute(self, image: np.ndarray) -> float:
        # Your metric computation here
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(gray.mean())
        return score

# Use it
metric = MyCustomMetric()
score = metric.compute(img)
```

---

## 🧪 Example Use Cases

### 1. Super-Resolution Model Evaluation
```python
from lens import LENSAnalyzer

# Compare SR methods
analyzer = LENSAnalyzer(normalize=True)
results = analyzer.compute_from_csv('sr_results.csv', image_column='Enhanced')
analyzer.plot_results(plot_type='bar', output_path='sr_comparison.png')
```

### 2. Image Enhancement Pipeline Testing
```python
from lens import LENSAnalyzer

images = [
    'original.png',
    'denoised.png',
    'sharpened.png',
    'enhanced.png'
]

analyzer = LENSAnalyzer(metrics=['Noise_Estimate', 'Laplacian_Variance', 'Colorfulness'])
results = analyzer.compute_multiple_images(images)
analyzer.plot_results(plot_type='bar')
```

### 3. Dataset Quality Assessment
```python
from lens import LENSAnalyzer
import glob

# Analyze all images in a folder
images = glob.glob('dataset/*.png')

analyzer = LENSAnalyzer()
results = analyzer.compute_multiple_images(images)

# Find low-quality images
low_quality = results[results['BRISQUE'] > 50]
print(f"Found {len(low_quality)} low-quality images")
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/yourusername/lens.git
cd lens
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Style

We use `black` for code formatting:
```bash
black lens/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use LENS in your research, please cite:
```bibtex
@software{lens2024,
  title={LENS: Library for Evaluating No-reference image quality Scores},
  author={Fela, R. F.},
  year={2024},
  url={https://github.com/fransfela/lens},
  version={0.1.0}
}
```

---

## 🙏 Acknowledgments

- BRISQUE implementation based on [PIQ](https://github.com/photosynthesis-team/piq)
- Colorfulness metric from Hasler & Süsstrunk (2003)
- Inspired by image quality assessment research community

---

## 📞 Contact

- **Author**: Randy Frans Fela
- **Email**: randyrff@gmail.com
- **GitHub**: [@fransfela](https://github.com/fransfela)
- **Issues**: [GitHub Issues](https://github.com/fransfela/lens/issues)

---

## 🗺️ Roadmap

### Current Version (v0.1.0)
- ✅ 10 core no-reference metrics
- ✅ Single/multiple image analysis
- ✅ CSV batch processing
- ✅ Normalization support
- ✅ Barplot and heatmap visualization

### Future Versions
- 🔜 v0.2.0: Add NIQE, CLIPIQA, MANIQA (deep learning metrics)
- 🔜 v0.3.0: Face quality metrics (detection, landmarks, recognition)
- 🔜 v0.4.0: Full-reference metrics (PSNR, SSIM, LPIPS)
- 🔜 v0.5.0: Video quality assessment support
- 🔜 v1.0.0: Stable API, comprehensive documentation

---

## ⭐ Star History

If you find LENS useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/lens&type=Date)](https://star-history.com/#yourusername/lens&Date)
