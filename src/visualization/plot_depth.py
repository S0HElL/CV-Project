# src/visualization/plot_depth.py
"""
Visualization for depth maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_depth_map(depth, title="Depth Map", cmap='plasma', vmin=None, vmax=None,
                   save_path=None, show=True):
    """
    Plot depth map with colormap.
    
    Args:
        depth: Depth map (meters)
        title: Plot title
        cmap: Colormap name
        vmin: Minimum depth for colormap
        vmax: Maximum depth for colormap
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Use valid depths for color scale if not specified
    valid_depth = depth[depth > 0]
    if vmin is None and len(valid_depth) > 0:
        vmin = valid_depth.min()
    if vmax is None and len(valid_depth) > 0:
        vmax = np.percentile(valid_depth, 95)  # Use 95th percentile to avoid outliers
    
    plt.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Depth (meters)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved depth plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_depth_histogram(depth, title="Depth Distribution", bins=50, 
                         save_path=None, show=True):
    """
    Plot histogram of depth values.
    
    Args:
        depth: Depth map (meters)
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    valid_depth = depth[depth > 0]
    
    if len(valid_depth) > 0:
        plt.hist(valid_depth, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Depth (meters)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_depth = np.mean(valid_depth)
        median_depth = np.median(valid_depth)
        plt.axvline(mean_depth, color='r', linestyle='--', label=f'Mean: {mean_depth:.2f}m')
        plt.axvline(median_depth, color='g', linestyle='--', label=f'Median: {median_depth:.2f}m')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No valid depth values', ha='center', va='center')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved depth histogram to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_depth_with_image(image, depth, title="Depth Map", save_path=None, show=True):
    """
    Plot original image and depth map side by side.
    
    Args:
        image: Original grayscale image
        depth: Depth map (meters)
        title: Title for depth map
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot depth
    valid_depth = depth[(depth > 0) & (depth < 80)]  # Filter for visualization
    if len(valid_depth) > 0:
        vmin = 0
        vmax = 50  # Fixed max at 50 meters for better visualization
    else:
        vmin = None
        vmax = None
    
    im = axes[1].imshow(depth, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Depth (meters)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved depth with image to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()