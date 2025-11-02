"""
Example 2: Multiple Images Analysis with Visualization

This example shows how to analyze multiple images and create visualizations.
"""

from lens import LENSAnalyzer
import os

def main():
    print("="*60)
    print("LENS Example 2: Multiple Images Analysis")
    print("="*60)
    
    # List of images to analyze
    images = [
        'data/sample_image.png',
        'data/lowres.png',
        'data/enhanced.png',
    ]
    
    # Check if images exist
    for img in images:
        if not os.path.exists(img):
            print(f"⚠️  Warning: {img} not found. Using placeholder.")
    
    # Method 1: Analyze with raw scores
    print("\n📊 Method 1: Raw scores (no normalization)")
    analyzer = LENSAnalyzer(normalize=False)
    results = analyzer.compute_multiple_images(images)
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    # Save results
    analyzer.save_results('results_raw.csv')
    
    # Method 2: Analyze with normalized scores
    print("\n📊 Method 2: Normalized scores (0-1 range)")
    analyzer = LENSAnalyzer(normalize=True)
    results = analyzer.compute_multiple_images(images)
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    # Save results
    analyzer.save_results('results_normalized.csv')
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # Barplot
    analyzer.plot_results(
        plot_type='bar',
        output_path='barplot.png',
        title='Image Quality Comparison'
    )
    print("  ✓ Saved: barplot.png")
    
    # Heatmap
    analyzer.plot_results(
        plot_type='heatmap',
        output_path='heatmap.png',
        title='Image Quality Metrics Heatmap'
    )
    print("  ✓ Saved: heatmap.png")
    
    # Method 3: Analyze specific metrics only
    print("\n📊 Method 3: Selected metrics only")
    analyzer = LENSAnalyzer(
        metrics=['BRISQUE', 'Laplacian_Variance', 'Colorfulness'],
        normalize=True
    )
    results = analyzer.compute_multiple_images(images)
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    analyzer.plot_results(plot_type='bar', output_path='barplot_selected.png')
    print("  ✓ Saved: barplot_selected.png")
    
    print("\n✓ Done! Check the output files.")

if __name__ == "__main__":
    main()