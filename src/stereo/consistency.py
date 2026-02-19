import numpy as np
import sys
import os

def compute_lr_consistency(disp_left, disp_right, threshold=1.5):
    """
    Vectorized Left-Right Consistency Check.
    
    Verifies that the disparity at pixel (x,y) in the left image matches
    the disparity at the corresponding point in the right image.
    
    Args:
        disp_left (np.array): Left-to-Right disparity map.
        disp_right (np.array): Right-to-Left disparity map.
        threshold (float): Max allowed difference. Increased to 1.5 for better coverage.
        
    Returns:
        filtered_disp (np.array): Consistency-checked disparity map.
        mask (np.array): Boolean mask of valid pixels.
    """
    height, width = disp_left.shape

    # 1. Create a grid of coordinates
    # We only need X coordinates, Y coordinates stay the same
    x_coords = np.arange(width)
    x_grid, y_grid = np.meshgrid(x_coords, np.arange(height))

    # 2. Calculate corresponding X in the right image: x_right = x_left - d_left
    # We use np.round() to find the nearest integer pixel, reducing quantization error
    x_right_locs = np.round(x_grid - disp_left).astype(np.int32)

    # 3. Create a mask for pixels that land INSIDE the right image
    # We also ignore pixels where disparity is 0 or negative (invalid)
    valid_locs = (x_right_locs >= 0) & \
                 (x_right_locs < width) & \
                 (disp_left > 0.0)

    # 4. Create an array to hold the right-image disparities
    # Initialize with a value that will definitely fail the check (e.g., infinity)
    disp_right_sampled = np.full_like(disp_left, fill_value=np.inf)

    # Only sample from valid locations to avoid IndexErrors
    # We use y_grid and x_right_locs to pull values from disp_right
    disp_right_sampled[valid_locs] = disp_right[y_grid[valid_locs], x_right_locs[valid_locs]]

    # 5. The Consistency Check
    # Calculate absolute difference
    diff = np.abs(disp_left - disp_right_sampled)

    # 6. Generate final mask
    # Valid if: location is valid AND difference is within threshold
    consistency_mask = valid_locs & (diff <= threshold)

    # 7. Apply mask
    filtered_disp = np.copy(disp_left)
    filtered_disp[~consistency_mask] = 0.0  # Set invalid pixels to 0

    return filtered_disp, consistency_mask
