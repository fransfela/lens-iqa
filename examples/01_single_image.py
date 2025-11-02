"""
Example 1: Single Image Analysis

This example shows how to compute all metrics for a single image.
"""

from lens import LENSAnalyzer
import cv2

def main():
    print("="*60)
    print("LENS Example 1: Single Image Analysis")
    print("="*60)
    
    # Path to your image
    image_path = 'data/sample_image.png'
    
    # Method 1: Using LENSAnalyzer (all metrics)
    print("\n📊 Method 1: Compute all metrics")
    analyzer = LENSAnalyzer()
    scores = analyzer.compute_single_image(image_path)
    
    print("\nResults:")
    for metric, score in scores.items():
        if score is not None:
            print(f"  {metric:25s}: {score:.4f}")
        else:
            print(f"  {metric:25s}: N/A")
    
    # Method 2: Using LENSAnalyzer (selected metrics)
    print("\n📊 Method 2: Compute selected metrics")
    analyzer = LENSAnalyzer(metrics=['BRISQUE', 'Laplacian_Variance', 'Colorfulness'])
    scores = analyzer.compute_single_image(image_path)
    
    print("\nResults:")
    for metric, score in scores.items():
        if score is not None:
            print(f"  {metric:25s}: {score:.4f}")
    
    # Method 3: Using individual metric classes
    print("\n📊 Method 3: Use individual metrics")
    from lens.metrics import LaplacianVariance, Colorfulness, NoiseEstimate
    
    img = cv2.imread(image_path)
    
    sharpness = LaplacianVariance().compute(img)
    color = Colorfulness().compute(img)
    noise = NoiseEstimate().compute(img)
    
    print(f"\n  Sharpness (Laplacian): {sharpness:.4f}")
    print(f"  Colorfulness:          {color:.4f}")
    print(f"  Noise Estimate:        {noise:.4f}")
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()