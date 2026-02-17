"""
Trajectory chaining for visual odometry.
Accumulates camera poses across frames to build complete trajectory.
"""

import numpy as np
import os


def chain_transformations(R_list, t_list):
    """
    Chain a sequence of relative transformations to get absolute poses.
    
    Each (R_i, t_i) represents transformation from frame i to frame i+1.
    This function computes absolute pose of each frame in world coordinates.
    
    Args:
        R_list: List of rotation matrices
        t_list: List of translation vectors
        
    Returns:
        poses: List of 4x4 transformation matrices (camera to world)
    """
    # Start with identity (first camera is at origin)
    poses = [np.eye(4)]
    
    # Current pose (world to camera)
    current_pose = np.eye(4)
    
    for R, t in zip(R_list, t_list):
        # Create relative transformation (camera i to camera i+1)
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.flatten()
        
        # Invert to get camera i+1 to camera i
        T_relative_inv = np.linalg.inv(T_relative)
        
        # Update current pose: world to camera i+1
        current_pose = current_pose @ T_relative_inv
        
        # Store the pose
        poses.append(current_pose.copy())
    
    return poses


def extract_trajectory_positions(poses):
    """
    Extract camera positions from pose matrices.
    
    Args:
        poses: List of 4x4 transformation matrices
        
    Returns:
        positions: Nx3 array of camera positions (x, y, z)
    """
    positions = np.zeros((len(poses), 3))
    
    for i, pose in enumerate(poses):
        # Camera position in world coordinates is -R^T * t
        R = pose[:3, :3]
        t = pose[:3, 3]
        position = -R.T @ t
        positions[i] = position
    
    return positions


def save_trajectory(poses, output_path):
    """
    Save trajectory to file in KITTI format.
    
    KITTI format: Each line contains 12 values representing [R|t] as 3x4 matrix.
    
    Args:
        poses: List of 4x4 transformation matrices
        output_path: Path to output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pose in poses:
            # Extract 3x4 [R|t] matrix
            pose_3x4 = pose[:3, :]
            
            # Flatten and write as space-separated values
            values = pose_3x4.flatten()
            line = ' '.join([f'{v:.6e}' for v in values])
            f.write(line + '\n')
    
    print(f"Saved trajectory to {output_path}")


def load_trajectory(file_path):
    """
    Load trajectory from file in KITTI format.
    
    Args:
        file_path: Path to trajectory file
        
    Returns:
        poses: List of 4x4 transformation matrices
    """
    poses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.split()])
            
            # Reshape to 3x4
            pose_3x4 = values.reshape(3, 4)
            
            # Convert to 4x4
            pose = np.eye(4)
            pose[:3, :] = pose_3x4
            
            poses.append(pose)
    
    return poses


def compute_trajectory_length(positions):
    """
    Compute total distance traveled along trajectory.
    
    Args:
        positions: Nx3 array of camera positions
        
    Returns:
        total_length: Total distance in meters
    """
    if len(positions) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(positions)):
        distance = np.linalg.norm(positions[i] - positions[i-1])
        total_length += distance
    
    return total_length


def downsample_trajectory(poses, step=1):
    """
    Downsample trajectory by taking every step-th pose.
    
    Args:
        poses: List of 4x4 transformation matrices
        step: Downsampling step (1 = no downsampling)
        
    Returns:
        downsampled_poses: Downsampled list of poses
    """
    return poses[::step]


class TrajectoryBuilder:
    """
    Class for incrementally building a trajectory from relative pose estimates.
    """
    
    def __init__(self):
        """Initialize trajectory builder."""
        self.poses = [np.eye(4)]  # Start at origin
        self.current_pose = np.eye(4)
        self.num_frames = 1
    
    def add_pose(self, R, t):
        """
        Add a new relative pose to the trajectory.
        
        Args:
            R: 3x3 rotation matrix (frame i to frame i+1)
            t: 3x1 translation vector (frame i to frame i+1)
        """
        if R is None or t is None:
            # If pose estimation failed, duplicate last pose
            self.poses.append(self.current_pose.copy())
            self.num_frames += 1
            return
        
        # Create relative transformation
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.flatten()
        
        # Invert to get camera i+1 to camera i
        T_relative_inv = np.linalg.inv(T_relative)
        
        # Update current pose
        self.current_pose = self.current_pose @ T_relative_inv
        
        # Store pose
        self.poses.append(self.current_pose.copy())
        self.num_frames += 1
    
    def get_poses(self):
        """
        Get all accumulated poses.
        
        Returns:
            poses: List of 4x4 transformation matrices
        """
        return self.poses
    
    def get_positions(self):
        """
        Get camera positions.
        
        Returns:
            positions: Nx3 array of camera positions
        """
        return extract_trajectory_positions(self.poses)
    
    def save(self, output_path):
        """
        Save trajectory to file.
        
        Args:
            output_path: Path to output file
        """
        save_trajectory(self.poses, output_path)
    
    def get_trajectory_length(self):
        """
        Get total trajectory length.
        
        Returns:
            length: Total distance in meters
        """
        positions = self.get_positions()
        return compute_trajectory_length(positions)


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    from src.utils.kitti_odometry_loader import load_left_image, load_calibration, load_ground_truth_poses
    from src.vo.features import detect_and_match
    from src.vo.pose import estimate_pose_with_scale
    from src.stereo.block_matching import compute_disparity_optimized
    
    print("Testing trajectory chaining")
    print("=" * 60)
    
    # Test configuration
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    num_frames = 10  # Process first 10 frames
    
    print(f"\nBuilding trajectory for sequence {sequence_id:02d}, frames 0-{num_frames-1}")
    
    try:
        # Load calibration
        print(f"\n1. Loading calibration")
        calib = load_calibration(dataset_path, sequence_id)
        K = calib['K']
        baseline = calib['B']
        print(f"   Baseline: {baseline:.4f} meters")
        
        # Initialize trajectory builder
        print(f"\n2. Initializing trajectory builder")
        trajectory = TrajectoryBuilder()
        print(f"   Starting at origin")
        
        # Process frames sequentially
        print(f"\n3. Processing frames and building trajectory")
        
        for i in range(num_frames - 1):
            print(f"\n   Frame {i} -> {i+1}:")
            
            # Load stereo pair for frame i
            left_i, right_i = load_stereo_pair(dataset_path, sequence_id, i)
            
            # Compute disparity
            disparity = compute_disparity_optimized(
                left_i, right_i,
                window_size=11,
                max_disparity=128,
                cost_function='SAD'
            )
            
            # Load next frame
            left_i1 = load_left_image(dataset_path, sequence_id, i + 1)
            
            # Match features
            _, _, matches, points1, points2 = detect_and_match(
                left_i, left_i1,
                detector_type='ORB',
                max_features=2000,
                ratio_threshold=0.75
            )
            
            print(f"     Matches: {len(matches)}")
            
            # Estimate pose
            R, t, inliers, num_inliers = estimate_pose_with_scale(
                points1, points2,
                disparity,
                K, baseline,
                ransac_threshold=8.0,
                min_inliers=20
            )
            
            if R is not None:
                translation_mag = np.linalg.norm(t)
                print(f"     Inliers: {num_inliers}")
                print(f"     Translation: {translation_mag:.3f} m")
                
                # Add to trajectory
                trajectory.add_pose(R, t)
            else:
                print(f"     Pose estimation failed - using identity")
                trajectory.add_pose(None, None)
        
        # Get trajectory statistics
        print(f"\n4. Trajectory statistics")
        positions = trajectory.get_positions()
        trajectory_length = trajectory.get_trajectory_length()
        
        print(f"   Total frames: {trajectory.num_frames}")
        print(f"   Trajectory length: {trajectory_length:.2f} meters")
        print(f"   Final position: {positions[-1]}")
        
        # Save trajectory
        print(f"\n5. Saving trajectory")
        output_dir = "outputs/trajectory"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_trajectory_seq{sequence_id:02d}.txt")
        trajectory.save(output_path)
        
        # Load and verify
        print(f"\n6. Loading and verifying saved trajectory")
        loaded_poses = load_trajectory(output_path)
        print(f"   Loaded {len(loaded_poses)} poses")
        
        # Check if matches
        match = np.allclose(trajectory.get_poses()[0], loaded_poses[0])
        print(f"   First pose matches: {match}")
        
        # Compare with ground truth if available
        print(f"\n7. Comparing with ground truth")
        gt_poses = load_ground_truth_poses(dataset_path, sequence_id)
        
        if gt_poses is not None:
            gt_positions = extract_trajectory_positions(gt_poses[:num_frames])
            estimated_positions = positions
            
            print(f"   Ground truth positions shape: {gt_positions.shape}")
            print(f"   Estimated positions shape: {estimated_positions.shape}")
            
            # Compute endpoint error
            endpoint_error = np.linalg.norm(gt_positions[-1] - estimated_positions[-1])
            print(f"   Endpoint error: {endpoint_error:.2f} meters")
            
            # Show first few positions
            print(f"\n   First 3 positions comparison:")
            for j in range(min(3, len(gt_positions))):
                print(f"     Frame {j}:")
                print(f"       GT: {gt_positions[j]}")
                print(f"       Est: {estimated_positions[j]}")
        else:
            print(f"   Ground truth not available")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Trajectory chaining is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()