from lens import LENSAnalyzer

# CSV with image paths
analyzer = LENSAnalyzer(
    metrics=['BRISQUE', 'Laplacian_Variance', 'Colorfulness'],
    normalize=True
)

results = analyzer.compute_from_csv(
    'dataset.csv', 
    image_column='Enhanced'  # Column name with image paths
)

analyzer.save_results('results_with_metrics.csv')
analyzer.plot_results(plot_type='heatmap')