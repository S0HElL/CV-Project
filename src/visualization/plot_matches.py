"""
Visualization for feature matches.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def plot_feature_matches(image1, image2, keypoints1, keypoints2, matches, 
                        inliers=None, max_display=100, title="Feature Matches",
                        save_path=None, show=True):
    """
    Plot feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of cv2.DMatch objects
        inliers: Boolean mask indicating inliers (optional)
        max_display: Maximum number of matches to display
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    # Limit number of matches
    if inliers is not None:
        # Show inliers in green, outliers in red
        inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
        outlier_matches = [m for i, m in enumerate(matches) if not inliers[i]]
        
        inlier_matches = inlier_matches[:max_display]
        outlier_matches = outlier_matches[:max(0, max_display - len(inlier_matches))]
        
        # Draw outliers first (red)
        if len(outlier_matches) > 0:
            vis_image = cv2.drawMatches(
                image1, keypoints1,
                image2, keypoints2,
                outlier_matches, None,
                matchColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        else:
            vis_image = np.hstack([image1, image2])
        
        # Draw inliers on top (green)
        if len(inlier_matches) > 0:
            vis_image = cv2.drawMatches(
                image1, keypoints1,
                image2, keypoints2,
                inlier_matches, vis_image,
                matchColor=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            )
    else:
        # Draw all matches in default color
        display_matches = matches[:max_display]
        vis_image = cv2.drawMatches(
            image1, keypoints1,
            image2, keypoints2,
            display_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    
    # Plot
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB) if len(vis_image.shape) == 3 else vis_image, cmap='gray')
    
    if inliers is not None:
        num_inliers = np.sum(inliers)
        title += f" (Inliers: {num_inliers}/{len(matches)})"
    else:
        title += f" ({len(matches)} matches)"
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved matches plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_matches_simple(image1, image2, points1, points2, inliers=None,
                       max_display=50, title="Feature Matches",
                       save_path=None, show=True):
    """
    Simple plot of matches using point coordinates.
    
    Args:
        image1: First image
        image2: Second image
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        inliers: Boolean mask indicating inliers (optional)
        max_display: Maximum number of matches to display
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Stack images horizontally
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if len(image1.shape) == 2:
        # Grayscale
        vis_image = np.zeros((max(h1, h2), w1 + w2), dtype=image1.dtype)
        vis_image[:h1, :w1] = image1
        vis_image[:h2, w1:w1+w2] = image2
    else:
        # Color
        vis_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=image1.dtype)
        vis_image[:h1, :w1] = image1
        vis_image[:h2, w1:w1+w2] = image2
    
    ax.imshow(vis_image, cmap='gray' if len(image1.shape) == 2 else None)
    
    # Offset points2 by image1 width
    points2_offset = points2.copy()
    points2_offset[:, 0] += w1
    
    # Select matches to display
    indices = np.arange(len(points1))
    if inliers is not None:
        inlier_indices = indices[inliers]
        outlier_indices = indices[~inliers]
        
        # Limit display
        inlier_indices = inlier_indices[:max_display]
        outlier_indices = outlier_indices[:max(0, max_display - len(inlier_indices))]
        
        # Draw outliers
        for idx in outlier_indices:
            ax.plot([points1[idx, 0], points2_offset[idx, 0]],
                   [points1[idx, 1], points2_offset[idx, 1]],
                   'r-', linewidth=0.5, alpha=0.5)
        
        # Draw inliers
        for idx in inlier_indices:
            ax.plot([points1[idx, 0], points2_offset[idx, 0]],
                   [points1[idx, 1], points2_offset[idx, 1]],
                   'g-', linewidth=1.0)
        
        title += f" (Inliers: {len(inlier_indices)}/{len(points1)})"
    else:
        indices = indices[:max_display]
        for idx in indices:
            ax.plot([points1[idx, 0], points2_offset[idx, 0]],
                   [points1[idx, 1], points2_offset[idx, 1]],
                   'b-', linewidth=0.5)
        
        title += f" ({len(indices)} matches)"
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved simple matches plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

