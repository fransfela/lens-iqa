from lens import LENSAnalyzer

# List of images
images = ['img1.png', 'img2.png', 'img3.png']

# Compute with raw scores
analyzer = LENSAnalyzer(normalize=False)
results = analyzer.compute_multiple_images(images)
print(results)

# Compute with normalized scores
analyzer = LENSAnalyzer(normalize=True)
results = analyzer.compute_multiple_images(images)
analyzer.save_results('results.csv')

# Generate plots
analyzer.plot_results(plot_type='bar', output_path='barplot.png')
analyzer.plot_results(plot_type='heatmap', output_path='heatmap.png')