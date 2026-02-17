"""
Semi-Global Block Matching using OpenCV.
"""

import numpy as np
import cv2


def compute_disparity_sgbm(left_image, right_image, min_disparity=0, num_disparities=128, block_size=11):
    """
    Compute disparity using Semi-Global Block Matching.
    
    Args:
        left_image: Left camera image (grayscale)
        right_image: Right camera image (grayscale)
        min_disparity: Minimum disparity
        num_disparities: Maximum disparity range (must be divisible by 16)
        block_size: Block size (must be odd, typically 3-11)
        
    Returns:
        disparity_map: Disparity map (float32, in pixels)
    """
    # Ensure num_disparities is divisible by 16
    num_disparities = ((num_disparities + 15) // 16) * 16
    
    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left_image, right_image)
    
    # Convert to float and scale (SGBM returns disparity * 16)
    disparity_map = disparity.astype(np.float32) / 16.0
    
    # Set negative disparities to 0
    disparity_map[disparity_map < 0] = 0
    
    return disparity_map