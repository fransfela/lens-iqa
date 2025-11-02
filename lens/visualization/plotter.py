"""
Visualization functions for LENS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from ..utils.metrics_info import METRICS_INFO

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10


def plot_barplot(df: pd.DataFrame, 
                 metrics: List[str],
                 output_path: Optional[str] = None,
                 figsize: tuple = (12, 6),
                 title: str = 'Image Quality Metrics'):
    """
    Create barplot for multiple images and metrics
    
    Args:
        df: DataFrame with 'Image' column and metric columns
        metrics: List of metric names to plot
        output_path: Path to save plot (optional)
        figsize: Figure size
        title: Plot title
    """
    # Prepare data for plotting
    image_names = df['Image'].values if 'Image' in df.columns else df.index
    n_images = len(image_names)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_images)
    width = 0.8 / n_metrics
    
    colors = sns.color_palette("Set2", n_metrics)
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        offset = (i - n_metrics/2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, 
                     label=metric, color=colors[i],
                     edgecolor='black', linewidth=1, alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Metric', fontsize=9, title_fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved barplot to: {output_path}")
    
    plt.show()


def plot_heatmap(df: pd.DataFrame,
                 metrics: List[str],
                 output_path: Optional[str] = None,
                 figsize: tuple = (10, 8),
                 title: str = 'Image Quality Metrics Heatmap'):
    """
    Create heatmap for multiple images and metrics
    
    Args:
        df: DataFrame with 'Image' column and metric columns
        metrics: List of metric names to plot
        output_path: Path to save plot (optional)
        figsize: Figure size
        title: Plot title
    """
    # Prepare data
    image_names = df['Image'].values if 'Image' in df.columns else df.index
    
    heatmap_data = []
    for metric in metrics:
        if metric in df.columns:
            heatmap_data.append(df[metric].values)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             columns=image_names,
                             index=metrics)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn',
               cbar_kws={'label': 'Score'},
               linewidths=0.5, linecolor='gray',
               ax=ax, annot_kws={'fontsize': 9})
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to: {output_path}")
    
    plt.show()