"""
Example 3: CSV Batch Processing

This example shows how to process a CSV file containing image paths
and add metric columns to the results.
"""

from lens import LENSAnalyzer
import pandas as pd
import os

def create_sample_csv():
    """Create a sample CSV file for demonstration"""
    data = {
        'Image': [
            'data/sample_image.png',
            'data/lowres.png',
            'data/enhanced.png'
        ],
        'Method': ['Original', 'Bicubic', 'SR_Model'],
        'Resolution': ['1920x1080', '640x480', '1920x1080']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_dataset.csv', index=False)
    print("✓ Created sample_dataset.csv")
    return 'sample_dataset.csv'

def main():
    print("="*60)
    print("LENS Example 3: CSV Batch Processing")
    print("="*60)
    
    # Create sample CSV (or use your own)
    csv_path = create_sample_csv()
    
    # Method 1: Process with all metrics
    print("\n📊 Method 1: Process with all metrics")
    analyzer = LENSAnalyzer(normalize=False)
    results = analyzer.compute_from_csv(csv_path, image_column='Image')
    
    print("\nResults (first 5 rows):")
    print(results.head().to_string(index=False))
    
    # Save results
    analyzer.save_results('results_with_metrics.csv')
    print("\n✓ Saved: results_with_metrics.csv")
    
    # Method 2: Process with selected metrics and normalization
    print("\n📊 Method 2: Selected metrics with normalization")
    analyzer = LENSAnalyzer(
        metrics=['BRISQUE', 'Laplacian_Variance', 'Colorfulness', 'Noise_Estimate'],
        normalize=True
    )
    results = analyzer.compute_from_csv(csv_path, image_column='Image')
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    # Save results
    analyzer.save_results('results_normalized.csv')
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # Barplot
    analyzer.plot_results(
        plot_type='bar',
        output_path='csv_barplot.png'
    )
    print("  ✓ Saved: csv_barplot.png")
    
    # Heatmap
    analyzer.plot_results(
        plot_type='heatmap',
        output_path='csv_heatmap.png'
    )
    print("  ✓ Saved: csv_heatmap.png")
    
    # Analyze results
    print("\n📈 Summary Statistics:")
    metric_cols = ['BRISQUE', 'Laplacian_Variance', 'Colorfulness', 'Noise_Estimate']
    for metric in metric_cols:
        if metric in results.columns:
            mean_val = results[metric].mean()
            std_val = results[metric].std()
            print(f"  {metric:25s}: {mean_val:.2f} ± {std_val:.2f}")
    
    print("\n✓ Done! Check the output files.")

if __name__ == "__main__":
    main()