"""
Metadata for all supported metrics in LENS
"""

METRICS_INFO = {
    'BRISQUE': {
        'full_name': 'Blind/Referenceless Image Spatial Quality Evaluator',
        'category': 'Quality',
        'description': 'Measures naturalness based on scene statistics. Detects blur, noise, and compression artifacts.',
        'range': (0, 100),
        'better': 'lower',
        'reference': 'Mittal et al., 2012',
        'paper_url': 'https://doi.org/10.1109/TIP.2012.2214050'
    },
    'PIQE': {
        'full_name': 'Perception-based Image Quality Evaluator',
        'category': 'Quality',
        'description': 'Block-based distortion estimation for perceptual quality assessment.',
        'range': (0, 100),
        'better': 'lower',
        'reference': 'Venkatanath et al., 2015',
        'paper_url': 'https://doi.org/10.1109/NCCI.2015.7084843'
    },
    'Laplacian_Variance': {
        'full_name': 'Laplacian Variance',
        'category': 'Sharpness',
        'description': 'Measures edge sharpness using second derivative (Laplacian operator).',
        'range': (0, 5000),  # Typical range
        'better': 'higher',
        'reference': 'Pech-Pacheco et al., 2000',
        'paper_url': None
    },
    'Tenengrad': {
        'full_name': 'Tenengrad Gradient',
        'category': 'Sharpness',
        'description': 'Focus measure based on gradient magnitude (Sobel operator).',
        'range': (0, float('inf')),
        'better': 'higher',
        'reference': 'Krotkov, 1988',
        'paper_url': None
    },
    'Brenner': {
        'full_name': 'Brenner Focus Measure',
        'category': 'Sharpness',
        'description': 'Fast focus measure based on horizontal gradient squared.',
        'range': (0, float('inf')),
        'better': 'higher',
        'reference': 'Brenner et al., 1976',
        'paper_url': None
    },
    'Edge_Density': {
        'full_name': 'Edge Density',
        'category': 'Sharpness',
        'description': 'Proportion of edge pixels detected using Canny edge detector.',
        'range': (0, 1),
        'better': 'higher',
        'reference': 'Canny, 1986',
        'paper_url': None
    },
    'Colorfulness': {
        'full_name': 'Colorfulness (Hasler & Süsstrunk)',
        'category': 'Color',
        'description': 'Perceptual colorfulness metric based on opponent color space.',
        'range': (0, 150),  # Typical range
        'better': 'higher',
        'reference': 'Hasler & Süsstrunk, 2003',
        'paper_url': 'https://doi.org/10.1117/12.477378'
    },
    'RMS_Contrast': {
        'full_name': 'RMS Contrast',
        'category': 'Color',
        'description': 'Root mean square contrast (standard deviation of pixel intensities).',
        'range': (0, 128),  # For 8-bit images
        'better': 'higher',
        'reference': 'Peli, 1990',
        'paper_url': None
    },
    'Michelson_Contrast': {
        'full_name': 'Michelson Contrast',
        'category': 'Color',
        'description': 'Contrast ratio: (L_max - L_min) / (L_max + L_min).',
        'range': (0, 1),
        'better': 'higher',
        'reference': 'Michelson, 1927',
        'paper_url': None
    },
    'Noise_Estimate': {
        'full_name': 'Noise Estimate (High-pass Filter)',
        'category': 'Noise',
        'description': 'Estimates noise level using Laplacian high-pass filter.',
        'range': (0, 50),  # Typical range
        'better': 'lower',
        'reference': 'Immerkaer, 1996',
        'paper_url': None
    }
}

# Optional/Future metrics (placeholders)
OPTIONAL_METRICS = {
    'NIQE': {
        'full_name': 'Natural Image Quality Evaluator',
        'category': 'Quality',
        'description': 'Compares to statistical model of natural images.',
        'range': (0, 20),
        'better': 'lower',
        'requires': ['pyiqa'],
        'reference': 'Mittal et al., 2013',
        'paper_url': 'https://doi.org/10.1109/LSP.2012.2227726'
    },
    'CLIPIQA': {
        'full_name': 'CLIP-based Image Quality Assessment',
        'category': 'Quality',
        'description': 'Deep learning-based quality assessment using CLIP features.',
        'range': (0, 1),
        'better': 'higher',
        'requires': ['pyiqa', 'torch'],
        'reference': 'Wang et al., 2022',
        'paper_url': None
    },
    'MANIQA': {
        'full_name': 'Multi-dimension Attention Network for IQA',
        'category': 'Quality',
        'description': 'State-of-the-art deep learning quality assessment.',
        'range': (0, 1),
        'better': 'higher',
        'requires': ['pyiqa', 'torch'],
        'reference': 'Yang et al., 2022',
        'paper_url': None
    }
}

def get_metrics_table():
    """Generate a formatted table of all metrics"""
    import pandas as pd
    
    data = []
    for metric_name, info in METRICS_INFO.items():
        data.append({
            'Metric': metric_name,
            'Full Name': info['full_name'],
            'Category': info['category'],
            'Range': f"{info['range'][0]}-{info['range'][1]}" if info['range'][1] != float('inf') else f"{info['range'][0]}-∞",
            'Better': info['better'].capitalize(),
            'Description': info['description'][:60] + '...' if len(info['description']) > 60 else info['description']
        })
    
    return pd.DataFrame(data)

def print_metrics_info():
    """Print detailed information about all metrics"""
    print("="*80)
    print("LENS - Available Image Quality Metrics")
    print("="*80)metrics_info.py
    
    for metric_name, info in METRICS_INFO.items():
        print(f"\n📊 {metric_name}")
        print(f"   Full Name: {info['full_name']}")
        print(f"   Category: {info['category']}")
        print(f"   Range: {info['range'][0]}-{info['range'][1]}")
        print(f"   Better: {info['better']}")
        print(f"   Description: {info['description']}")
        if info['reference']:
            print(f"   Reference: {info['reference']}")