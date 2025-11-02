# Changelog

All notable changes to LENS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-02

### Added
- Initial release of LENS
- 10 no-reference image quality metrics:
  - Quality: BRISQUE, PIQE
  - Sharpness: Laplacian Variance, Tenengrad, Brenner, Edge Density
  - Color: Colorfulness, RMS Contrast, Michelson Contrast
  - Noise: Noise Estimate
- Single image analysis
- Multiple images analysis with batch processing
- CSV batch processing support
- Optional score normalization to 0-1 range
- Scientific visualization (barplot and heatmap)
- Comprehensive documentation and examples
- Unit tests for core functionality

### Dependencies
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- torch >= 1.9.0 (optional, for BRISQUE)
- piq >= 0.7.0 (optional, for BRISQUE)

## [Unreleased]

### Planned for v0.2.0
- [ ] Add NIQE metric (deep learning-based)
- [ ] Add CLIPIQA metric (deep learning-based)
- [ ] Add MANIQA metric (deep learning-based)
- [ ] Improved documentation
- [ ] More visualization options

### Planned for v0.3.0
- [ ] Face quality metrics
  - [ ] Face detection confidence
  - [ ] Facial landmark quality
  - [ ] Face recognition embedding quality
- [ ] GPU acceleration support
- [ ] Progress bars for batch processing

### Planned for v0.4.0
- [ ] Full-reference metrics (PSNR, SSIM, LPIPS)
- [ ] Video quality assessment
- [ ] Real-time processing support

### Planned for v1.0.0
- [ ] Stable API
- [ ] Complete documentation
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance benchmarks
- [ ] CLI tool