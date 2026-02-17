"""
Visualization for disparity maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def plot_disparity_map(disparity, title="Disparity Map", cmap='jet', vmin=None, vmax=None, 
                       save_path=None, show=True):
    """
    Plot disparity map with colormap.
    
    Args:
        disparity: Disparity map
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Use valid disparities for color scale if not specified
    valid_disparity = disparity[disparity > 0]
    if vmin is None and len(valid_disparity) > 0:
        vmin = valid_disparity.min()
    if vmax is None and len(valid_disparity) > 0:
        vmax = valid_disparity.max()
    
    plt.imshow(disparity, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Disparity (pixels)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved disparity plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_disparity_comparison(disparity1, disparity2, title1="Disparity 1", 
                              title2="Disparity 2", save_path=None, show=True):
    """
    Plot two disparity maps side by side for comparison.
    
    Args:
        disparity1: First disparity map
        disparity2: Second disparity map
        title1: Title for first map
        title2: Title for second map
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute common color scale
    valid1 = disparity1[disparity1 > 0]
    valid2 = disparity2[disparity2 > 0]
    
    if len(valid1) > 0 and len(valid2) > 0:
        vmin = min(valid1.min(), valid2.min())
        vmax = max(valid1.max(), valid2.max())
    else:
        vmin, vmax = None, None
    
    # Plot first disparity
    im1 = axes[0].imshow(disparity1, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(title1)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Disparity (pixels)')
    
    # Plot second disparity
    im2 = axes[1].imshow(disparity2, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Disparity (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved disparity comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_disparity_with_image(image, disparity, title="Disparity", save_path=None, show=True):
    """
    Plot original image and disparity map side by side.
    
    Args:
        image: Original grayscale image
        disparity: Disparity map
        title: Title for the plot
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot disparity
    valid_disparity = disparity[disparity > 0]
    vmin = valid_disparity.min() if len(valid_disparity) > 0 else None
    vmax = valid_disparity.max() if len(valid_disparity) > 0 else None
    
    im = axes[1].imshow(disparity, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Disparity (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved disparity with image to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()