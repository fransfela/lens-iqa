from lens import LENSAnalyzer

# Compute all metrics for single image
analyzer = LENSAnalyzer()
scores = analyzer.compute_single_image('image.png')
print(scores)

# Compute specific metrics
analyzer = LENSAnalyzer(metrics=['BRISQUE', 'Laplacian_Variance'])
scores = analyzer.compute_single_image('image.png')