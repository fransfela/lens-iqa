# LENS Examples

This directory contains example scripts demonstrating various use cases of LENS.

## 📁 Files

- **`01_single_image.py`** - Analyze a single image with all or selected metrics
- **`02_multiple_images.py`** - Analyze multiple images and create visualizations
- **`03_csv_batch.py`** - Batch process images from a CSV file
- **`quickstart.ipynb`** - Jupyter notebook tutorial with interactive examples

## 🚀 Running Examples

### 1. Single Image Analysis
```bash
python 01_single_image.py
```

Demonstrates:
- Computing all metrics for one image
- Computing selected metrics only
- Using individual metric classes

### 2. Multiple Images Analysis
```bash
python 02_multiple_images.py
```

Demonstrates:
- Analyzing multiple images
- Raw vs normalized scores
- Creating barplots and heatmaps
- Saving results to CSV

### 3. CSV Batch Processing
```bash
python 03_csv_batch.py
```

Demonstrates:
- Processing images from CSV file
- Adding metric columns to existing data
- Creating visualizations from CSV results
- Computing summary statistics

### 4. Jupyter Notebook Tutorial
```bash
jupyter notebook quickstart.ipynb
```

Interactive tutorial covering all basic features.

## 📊 Sample Data

The examples expect sample images in the `data/` subdirectory:
- `sample_image.png` - Original image
- `lowres.png` - Low resolution version
- `enhanced.png` - Enhanced/super-resolved version

You can replace these with your own images.

## 💡 Tips

1. **Start with `01_single_image.py`** if you're new to LENS
2. **Use normalization** when comparing multiple images: `normalize=True`
3. **Select specific metrics** to speed up computation
4. **Check metric directions** in the documentation (some are "lower is better")

## 🔗 More Information

- [Full Documentation](../README.md)
- [API Reference](../docs/api_reference.md)
- [Metrics Guide](../docs/metrics_guide.md)