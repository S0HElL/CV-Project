"""
KITTI odometry dataset loader.
Loads monocular images, calibration data, and ground truth poses.
"""

import os
import numpy as np
import cv2
import yaml

def load_left_image(dataset_path, sequence_id, frame_id):
    """
    Load left camera image for visual odometry.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        frame_id: Frame number within sequence
        
    Returns:
        image: Left camera image as numpy array (grayscale)
    """
    sequence_str = f"{sequence_id:02d}"
    
    image_path = os.path.join(
        dataset_path, 
        "sequences", 
        sequence_str, 
        "image_0", 
        f"{frame_id:06d}.png"
    )
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return image


def load_calibration(dataset_path, sequence_id):
    """
    Load calibration parameters for a sequence.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        
    Returns:
        calibration: Dictionary containing:
            - 'f': focal length (pixels)
            - 'B': baseline (meters)
            - 'K': intrinsic matrix (3x3)
            - 'P0': projection matrix for camera 0 (3x4)
            - 'P1': projection matrix for camera 1 (3x4)
    """
    sequence_str = f"{sequence_id:02d}"
    
    calib_path = os.path.join(
        dataset_path, 
        "sequences", 
        sequence_str, 
        "calib.txt"
    )
    
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    
    # Parse calibration file
    calib_data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if line.strip():
                key, values = line.split(':', 1)
                calib_data[key.strip()] = np.array([float(x) for x in values.split()])
    
    # Extract P0 (left camera projection matrix) and P1 (right camera projection matrix)
    P0 = calib_data['P0'].reshape(3, 4)
    P1 = calib_data['P1'].reshape(3, 4)
    
    # Extract intrinsic matrix K from P0
    K = P0[:3, :3]
    
    # Focal length (assuming fx = fy, using fx)
    f = K[0, 0]
    
    # Baseline: B = -P1[0, 3] / P1[0, 0]
    # P1[0, 3] = -f * B (for rectified stereo)
    B = -P1[0, 3] / P1[0, 0]
    
    calibration = {
        'f': f,
        'B': B,
        'K': K,
        'P0': P0,
        'P1': P1,
        'cx': K[0, 2],
        'cy': K[1, 2]
    }
    
    return calibration


def load_ground_truth_poses(dataset_path, sequence_id):
    """
    Load ground truth camera poses for a sequence.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        
    Returns:
        poses: List of 4x4 transformation matrices (camera to world)
               Each pose is a homogeneous transformation matrix
    """
    sequence_str = f"{sequence_id:02d}"
    
    poses_path = os.path.join(
        dataset_path, 
        "poses", 
        f"{sequence_str}.txt"
    )
    
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Ground truth poses not found: {poses_path}")
    
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            if line.strip():
                # Each line contains 12 values representing a 3x4 matrix [R|t]
                values = np.array([float(x) for x in line.split()])
                
                # Reshape to 3x4
                pose_3x4 = values.reshape(3, 4)
                
                # Convert to 4x4 homogeneous matrix
                pose = np.eye(4)
                pose[:3, :] = pose_3x4
                
                poses.append(pose)
    
    return poses


def get_sequence_info(dataset_path, sequence_id):
    """
    Get information about a sequence.
    
    Args:
        dataset_path: Path to KITTI odometry dataset root
        sequence_id: Sequence number (0-10)
        
    Returns:
        info: Dictionary containing sequence information
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
    
    # Count frames
    frames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    num_frames = len(frames)
    
    # Load calibration
    calib = load_calibration(dataset_path, sequence_id)
    
    # Check if ground truth exists
    poses_path = os.path.join(dataset_path, "poses", f"{sequence_str}.txt")
    has_ground_truth = os.path.exists(poses_path)
    
    info = {
        'sequence_id': sequence_id,
        'num_frames': num_frames,
        'has_ground_truth': has_ground_truth,
        'focal_length': calib['f'],
        'baseline': calib['B']
    }
    
    return info

if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_path = config['dataset']['kitti_odometry_path']
    sequence_id = 0
    frame_id = 0
    
    left_image = load_left_image(dataset_path, sequence_id, frame_id)
    calibration = load_calibration(dataset_path, sequence_id)
    ground_truth_poses = load_ground_truth_poses(dataset_path, sequence_id)
    
    print(f"Loaded left image shape: {left_image.shape}")
    print(f"Calibration focal length: {calibration['f']}, baseline: {calibration['B']}")
    print(f"Number of ground truth poses: {len(ground_truth_poses)}")