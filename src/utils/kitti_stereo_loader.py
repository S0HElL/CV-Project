"""
KITTI stereo dataset loader.
Loads stereo image pairs from KITTI odometry dataset.
"""

import os
import numpy as np
import cv2
import yaml


def load_stereo_pair(dataset_path, sequence_id, frame_id):
    """
    Load left and right stereo images for a given sequence and frame.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        frame_id: Frame number within sequence
        
    Returns:
        left_image: Left camera image as numpy array (grayscale)
        right_image: Right camera image as numpy array (grayscale)
    """
    sequence_str = f"{sequence_id:02d}"
    
    # KITTI odometry structure: sequences/XX/image_0 (left), image_1 (right)
    left_path = os.path.join(
        dataset_path, 
        "sequences", 
        sequence_str, 
        "image_0", 
        f"{frame_id:06d}.png"
    )
    
    right_path = os.path.join(
        dataset_path, 
        "sequences", 
        sequence_str, 
        "image_1", 
        f"{frame_id:06d}.png"
    )
    
    # Load images as grayscale
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if left_image is None:
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if right_image is None:
        raise FileNotFoundError(f"Right image not found: {right_path}")
    
    return left_image, right_image


def get_sequence_length(dataset_path, sequence_id):
    """
    Get the number of frames in a sequence.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        
    Returns:
        num_frames: Number of frames in the sequence
    """
    sequence_str = f"{sequence_id:02d}"
    image_dir = os.path.join(
        dataset_path, 
        "sequences", 
        sequence_str, 
        "image_0"
    )
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Sequence directory not found: {image_dir}")
    
    # Count PNG files
    frames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    return len(frames)

if __name__ == "__main__":
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_path = config['dataset']['kitti_odometry_path']
    sequence_id = 0
    frame_id = 0
    try:
        left, right = load_stereo_pair(dataset_path, sequence_id, frame_id)
        print(f"Loaded stereo pair for sequence {sequence_id}, frame {frame_id}")
        print(f"Left image shape: {left.shape}")
        print(f"Right image shape: {right.shape}")
    except FileNotFoundError as e:
        print(e)