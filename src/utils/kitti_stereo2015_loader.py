"""
KITTI Stereo 2015 dataset loader.
Loads stereo images and disparity ground truth.
"""

import os
import numpy as np
import cv2
from PIL import Image


def load_stereo_2015_pair(dataset_path, image_id):
    """
    Load stereo pair from KITTI Stereo 2015.
    
    Args:
        dataset_path: Path to KITTI Stereo 2015 training folder
        image_id: Image ID (e.g., 0, 1, 2, ..., 199)
        
    Returns:
        left_image: Left camera image (grayscale)
        right_image: Right camera image (grayscale)
    """
    left_path = os.path.join(dataset_path, "image_2", f"{image_id:06d}_10.png")
    right_path = os.path.join(dataset_path, "image_3", f"{image_id:06d}_10.png")
    
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if left_image is None or right_image is None:
        raise FileNotFoundError(f"Images not found for ID {image_id}")
    
    return left_image, right_image


def load_disparity_ground_truth(dataset_path, image_id, occluded=False):
    """
    Load disparity ground truth from KITTI Stereo 2015.
    
    Args:
        dataset_path: Path to KITTI Stereo 2015 training folder
        image_id: Image ID
        occluded: If True, load occluded disparity, else non-occluded
        
    Returns:
        disparity_gt: Ground truth disparity map (float32, in pixels)
    """
    folder = "disp_occ_0" if occluded else "disp_noc_0"
    disp_path = os.path.join(dataset_path, folder, f"{image_id:06d}_10.png")
    
    # Load disparity as 16-bit PNG
    disp_img = Image.open(disp_path)
    disp_array = np.array(disp_img, dtype=np.float32)
    
    # KITTI stores disparity as uint16 with scale factor 256
    disparity_gt = disp_array / 256.0
    
    return disparity_gt


def get_stereo_2015_calibration():
    """
    Get calibration for KITTI Stereo 2015.
    
    Note: All Stereo 2015 images use the same calibration.
    
    Returns:
        calibration: Dictionary with focal length and baseline
    """
    # Standard KITTI Stereo 2015 calibration
    focal_length = 721.5377  # pixels
    baseline = 0.5372  # meters
    cx = 609.5593
    cy = 172.8540
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    return {
        'f': focal_length,
        'B': baseline,
        'K': K,
        'cx': cx,
        'cy': cy
    }


def get_num_stereo_2015_images(dataset_path):
    """
    Get number of images in KITTI Stereo 2015 training set.
    
    Args:
        dataset_path: Path to KITTI Stereo 2015 training folder
        
    Returns:
        num_images: Number of images (typically 200)
    """
    image_dir = os.path.join(dataset_path, "image_2")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    images = [f for f in os.listdir(image_dir) if f.endswith('_10.png')]
    return len(images)