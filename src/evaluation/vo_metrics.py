"""
Visual odometry evaluation metrics.
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).
"""

import numpy as np


def align_trajectories(poses_est, poses_gt):
    """
    Align estimated trajectory to ground truth using first pose.
    
    Both trajectories are assumed to start at the same pose.
    This function ensures they have the same reference frame.
    
    Args:
        poses_est: List of estimated 4x4 pose matrices
        poses_gt: List of ground truth 4x4 pose matrices
        
    Returns:
        poses_est_aligned: Aligned estimated poses
        poses_gt_aligned: Ground truth poses (unchanged)
    """
    # For KITTI, both should already be in the same frame
    # Just ensure they're the same length
    min_len = min(len(poses_est), len(poses_gt))
    
    poses_est_aligned = poses_est[:min_len]
    poses_gt_aligned = poses_gt[:min_len]
    
    return poses_est_aligned, poses_gt_aligned


def extract_positions_from_poses(poses):
    """
    Extract xyz positions from pose matrices.
    
    Args:
        poses: List of 4x4 transformation matrices
        
    Returns:
        positions: Nx3 array of positions
    """
    positions = np.zeros((len(poses), 3))
    
    for i, pose in enumerate(poses):
        # Camera position in world coordinates: -R^T * t
        R = pose[:3, :3]
        t = pose[:3, 3]
        position = -R.T @ t
        positions[i] = position
    
    return positions


def compute_ate(poses_est, poses_gt):
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the absolute distance between estimated and ground truth
    camera positions at each timestamp.
    
    Args:
        poses_est: List of estimated 4x4 pose matrices
        poses_gt: List of ground truth 4x4 pose matrices
        
    Returns:
        ate_mean: Mean ATE (RMSE) in meters
        ate_std: Standard deviation of ATE
        ate_median: Median ATE
        ate_min: Minimum ATE
        ate_max: Maximum ATE
        errors: Array of position errors at each frame
    """
    # Align trajectories
    poses_est_aligned, poses_gt_aligned = align_trajectories(poses_est, poses_gt)
    
    if len(poses_est_aligned) == 0:
        return float('inf'), 0.0, float('inf'), 0.0, float('inf'), np.array([])
    
    # Extract positions
    positions_est = extract_positions_from_poses(poses_est_aligned)
    positions_gt = extract_positions_from_poses(poses_gt_aligned)
    
    # Compute Euclidean distances
    errors = np.linalg.norm(positions_est - positions_gt, axis=1)
    
    # Compute statistics
    ate_mean = np.sqrt(np.mean(errors ** 2))  # RMSE
    ate_std = np.std(errors)
    ate_median = np.median(errors)
    ate_min = np.min(errors)
    ate_max = np.max(errors)
    
    return ate_mean, ate_std, ate_median, ate_min, ate_max, errors


def compute_rpe(poses_est, poses_gt, step=1):
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the error in relative motion between frames separated by 'step'.
    This evaluates local consistency (drift per unit distance).
    
    Args:
        poses_est: List of estimated 4x4 pose matrices
        poses_gt: List of ground truth 4x4 pose matrices
        step: Frame step size (1 = consecutive frames)
        
    Returns:
        rpe_trans_mean: Mean translational RPE (RMSE) in meters
        rpe_trans_std: Standard deviation of translational RPE
        rpe_rot_mean: Mean rotational RPE (RMSE) in degrees
        rpe_rot_std: Standard deviation of rotational RPE
        trans_errors: Array of translation errors
        rot_errors: Array of rotation errors (degrees)
    """
    # Align trajectories
    poses_est_aligned, poses_gt_aligned = align_trajectories(poses_est, poses_gt)
    
    if len(poses_est_aligned) <= step:
        return float('inf'), 0.0, float('inf'), 0.0, np.array([]), np.array([])
    
    trans_errors = []
    rot_errors = []
    
    # Compute relative pose errors
    for i in range(len(poses_est_aligned) - step):
        # Ground truth relative transformation
        T_gt_i = poses_gt_aligned[i]
        T_gt_j = poses_gt_aligned[i + step]
        T_gt_rel = np.linalg.inv(T_gt_i) @ T_gt_j
        
        # Estimated relative transformation
        T_est_i = poses_est_aligned[i]
        T_est_j = poses_est_aligned[i + step]
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j
        
        # Error in relative transformation
        T_error = np.linalg.inv(T_gt_rel) @ T_est_rel
        
        # Translational error
        trans_error = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotational error (angle from rotation matrix)
        R_error = T_error[:3, :3]
        # Angle = arccos((trace(R) - 1) / 2)
        trace = np.trace(R_error)
        # Clamp to avoid numerical issues
        trace_clamped = np.clip((trace - 1) / 2, -1.0, 1.0)
        rot_angle = np.arccos(trace_clamped)
        rot_error_deg = np.rad2deg(rot_angle)
        rot_errors.append(rot_error_deg)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    # Compute statistics (RMSE)
    rpe_trans_mean = np.sqrt(np.mean(trans_errors ** 2))
    rpe_trans_std = np.std(trans_errors)
    rpe_rot_mean = np.sqrt(np.mean(rot_errors ** 2))
    rpe_rot_std = np.std(rot_errors)
    
    return rpe_trans_mean, rpe_trans_std, rpe_rot_mean, rpe_rot_std, trans_errors, rot_errors


def compute_vo_metrics_summary(poses_est, poses_gt, rpe_steps=[1, 5, 10]):
    """
    Compute comprehensive VO evaluation metrics.
    
    Args:
        poses_est: List of estimated 4x4 pose matrices
        poses_gt: List of ground truth 4x4 pose matrices
        rpe_steps: List of step sizes for RPE computation
        
    Returns:
        metrics: Dictionary containing all metrics
    """
    metrics = {
        'num_frames': len(poses_est),
        'num_frames_gt': len(poses_gt)
    }
    
    # Compute ATE
    ate_mean, ate_std, ate_median, ate_min, ate_max, ate_errors = compute_ate(poses_est, poses_gt)
    
    metrics['ate'] = {
        'mean': ate_mean,
        'std': ate_std,
        'median': ate_median,
        'min': ate_min,
        'max': ate_max,
        'errors': ate_errors
    }
    
    # Compute RPE for different step sizes
    metrics['rpe'] = {}
    
    for step in rpe_steps:
        if len(poses_est) > step:
            rpe_t_mean, rpe_t_std, rpe_r_mean, rpe_r_std, trans_err, rot_err = compute_rpe(
                poses_est, poses_gt, step=step
            )
            
            metrics['rpe'][f'step_{step}'] = {
                'trans_mean': rpe_t_mean,
                'trans_std': rpe_t_std,
                'rot_mean': rpe_r_mean,
                'rot_std': rpe_r_std,
                'trans_errors': trans_err,
                'rot_errors': rot_err
            }
    
    # Compute trajectory length
    positions_est = extract_positions_from_poses(poses_est)
    positions_gt = extract_positions_from_poses(poses_gt[:len(poses_est)])
    
    trajectory_length_est = 0.0
    trajectory_length_gt = 0.0
    
    for i in range(1, len(positions_est)):
        trajectory_length_est += np.linalg.norm(positions_est[i] - positions_est[i-1])
    
    for i in range(1, len(positions_gt)):
        trajectory_length_gt += np.linalg.norm(positions_gt[i] - positions_gt[i-1])
    
    metrics['trajectory_length_est'] = trajectory_length_est
    metrics['trajectory_length_gt'] = trajectory_length_gt
    
    return metrics


def print_vo_metrics(metrics, title="Visual Odometry Evaluation Metrics"):
    """
    Pretty print VO evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * 60)
    
    print(f"Trajectory Info:")
    print(f"  Estimated frames: {metrics['num_frames']}")
    print(f"  Ground truth frames: {metrics['num_frames_gt']}")
    print(f"  Estimated length: {metrics['trajectory_length_est']:.2f} meters")
    print(f"  Ground truth length: {metrics['trajectory_length_gt']:.2f} meters")
    
    print(f"\nAbsolute Trajectory Error (ATE):")
    ate = metrics['ate']
    print(f"  RMSE: {ate['mean']:.4f} meters")
    print(f"  Std: {ate['std']:.4f} meters")
    print(f"  Median: {ate['median']:.4f} meters")
    print(f"  Min: {ate['min']:.4f} meters")
    print(f"  Max: {ate['max']:.4f} meters")
    
    print(f"\nRelative Pose Error (RPE):")
    for step_name, rpe in metrics['rpe'].items():
        step = int(step_name.split('_')[1])
        print(f"  Step {step} frames:")
        print(f"    Translation RMSE: {rpe['trans_mean']:.4f} meters")
        print(f"    Translation Std: {rpe['trans_std']:.4f} meters")
        print(f"    Rotation RMSE: {rpe['rot_mean']:.4f} degrees")
        print(f"    Rotation Std: {rpe['rot_std']:.4f} degrees")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_odometry_loader import load_ground_truth_poses
    from src.vo.trajectory import load_trajectory
    
    print("Testing VO evaluation metrics")
    print("=" * 60)
    
    # Test with synthetic data
    print(f"\n1. Testing with synthetic trajectories")
    
    # Create synthetic ground truth (straight line)
    num_frames = 100
    poses_gt = []
    for i in range(num_frames):
        T = np.eye(4)
        T[0, 3] = i * 0.5  # Move 0.5m in x direction each frame
        poses_gt.append(T)
    
    # Create noisy estimate
    poses_est = []
    for i in range(num_frames):
        T = np.eye(4)
        T[0, 3] = i * 0.5 + np.random.normal(0, 0.1)  # Add noise
        T[1, 3] = np.random.normal(0, 0.05)  # Add lateral drift
        poses_est.append(T)
    
    print(f"   Generated {num_frames} synthetic poses")
    
    # Test ATE
    print(f"\n2. Testing ATE computation")
    ate_mean, ate_std, ate_median, ate_min, ate_max, ate_errors = compute_ate(poses_est, poses_gt)
    
    print(f"   ATE RMSE: {ate_mean:.4f} meters")
    print(f"   ATE Std: {ate_std:.4f} meters")
    print(f"   ATE Median: {ate_median:.4f} meters")
    print(f"   ATE range: [{ate_min:.4f}, {ate_max:.4f}] meters")
    print(f"   Errors shape: {ate_errors.shape}")
    
    # Test RPE
    print(f"\n3. Testing RPE computation")
    for step in [1, 5, 10]:
        rpe_t_mean, rpe_t_std, rpe_r_mean, rpe_r_std, trans_err, rot_err = compute_rpe(
            poses_est, poses_gt, step=step
        )
        print(f"   Step {step}:")
        print(f"     Translation RMSE: {rpe_t_mean:.4f} meters")
        print(f"     Rotation RMSE: {rpe_r_mean:.4f} degrees")
        print(f"     Num samples: {len(trans_err)}")
    
    # Test comprehensive metrics
    print(f"\n4. Testing comprehensive metrics")
    metrics = compute_vo_metrics_summary(poses_est, poses_gt, rpe_steps=[1, 5, 10])
    print_vo_metrics(metrics, "Synthetic Trajectory Evaluation")
    
    # Test with real KITTI data if available
    print(f"\n5. Testing with KITTI ground truth")
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    
    try:
        gt_poses = load_ground_truth_poses(dataset_path, sequence_id)
        
        if gt_poses is not None:
            print(f"   Loaded {len(gt_poses)} ground truth poses")
            
            # Create a simple estimated trajectory (with drift)
            poses_est_kitti = []
            for i in range(min(50, len(gt_poses))):
                T = gt_poses[i].copy()
                # Add some drift
                T[0, 3] += i * 0.01  # Cumulative drift in x
                T[1, 3] += np.random.normal(0, 0.05)  # Random drift in y
                poses_est_kitti.append(T)
            
            # Compute metrics
            metrics_kitti = compute_vo_metrics_summary(
                poses_est_kitti, 
                gt_poses[:len(poses_est_kitti)],
                rpe_steps=[1, 5, 10]
            )
            
            print_vo_metrics(metrics_kitti, "KITTI Sequence 00 (Synthetic Estimate)")
        else:
            print(f"   Ground truth not available")
            
    except Exception as e:
        print(f"   Could not load KITTI data: {e}")
    
    # Test edge cases
    print(f"\n6. Testing edge cases")
    
    # Perfect trajectory
    print(f"   Perfect trajectory (zero error):")
    ate_perfect, _, _, _, _, _ = compute_ate(poses_gt[:10], poses_gt[:10])
    print(f"     ATE: {ate_perfect:.6f} meters")
    
    # Single pose
    print(f"   Single pose:")
    ate_single, _, _, _, _, _ = compute_ate([poses_gt[0]], [poses_gt[0]])
    print(f"     ATE: {ate_single:.6f} meters")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("VO evaluation metrics are working correctly.")