#!/usr/bin/env python3
"""
Script to run visual odometry and compute camera trajectory.
Processes KITTI sequences and saves trajectories.
"""

import sys
import os
import yaml
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.kitti_stereo_loader import load_stereo_pair
from src.utils.kitti_odometry_loader import load_left_image, load_calibration
from src.stereo.block_matching import compute_disparity_optimized
from src.vo.features import detect_and_match
from src.vo.pose import estimate_pose_with_scale
from src.vo.trajectory import TrajectoryBuilder, extract_trajectory_positions
from src.visualization.plot_trajectory import plot_trajectory_2d, plot_trajectory_3d
from src.visualization.plot_matches import plot_feature_matches


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_vo_pipeline(dataset_path, sequence_id, num_frames, config, output_dir):
    """
    Run visual odometry pipeline on sequence.
    
    Args:
        dataset_path: Path to KITTI dataset
        sequence_id: Sequence number
        num_frames: Number of frames to process
        config: Configuration dictionary
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Visual Odometry - Sequence {sequence_id:02d}")
    print(f"{'='*60}")
    
    # Load calibration
    print(f"\nLoading calibration...")
    calib = load_calibration(dataset_path, sequence_id)
    K = calib['K']
    baseline = calib['B']
    print(f"  Focal length: {calib['f']:.2f} pixels")
    print(f"  Baseline: {baseline:.4f} meters")
    
    # VO parameters
    detector_type = config['vo']['feature_detector']
    ransac_threshold = config['vo']['ransac_threshold']
    
    print(f"\nVO parameters:")
    print(f"  Feature detector: {detector_type}")
    print(f"  RANSAC threshold: {ransac_threshold}")
    
    # Create output directories
    trajectory_dir = os.path.join(output_dir, config['output']['trajectory_dir'])
    vis_dir = os.path.join(output_dir, config['output']['visualizations_dir'], f"vo_seq_{sequence_id:02d}")
    
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize trajectory
    trajectory = TrajectoryBuilder()
    print(f"\nProcessing {num_frames} frames...")
    
    # Process frames
    for i in range(num_frames - 1):
        print(f"\n--- Frame {i} -> {i+1} ---")
        
        try:
            # Load stereo pair for frame i
            left_i, right_i = load_stereo_pair(dataset_path, sequence_id, i)
            
            # Compute disparity for frame i
            print(f"  Computing disparity...")
            disparity = compute_disparity_optimized(
                left_i, right_i,
                window_size=config['stereo']['window_size'],
                max_disparity=config['stereo']['max_disparity'],
                cost_function=config['stereo']['cost_function']
            )
            
            # Load frame i+1
            left_i1 = load_left_image(dataset_path, sequence_id, i + 1)
            
            # Detect and match features
            print(f"  Detecting and matching features...")
            kp1, kp2, matches, points1, points2 = detect_and_match(
                left_i, left_i1,
                detector_type=detector_type,
                max_features=3000,
                ratio_threshold=0.75
            )
            print(f"    Matches: {len(matches)}")
            
            # Estimate pose
            print(f"  Estimating pose...")
            R, t, inliers, num_inliers = estimate_pose_with_scale(
                points1, points2,
                disparity,
                K, baseline,
                ransac_threshold=ransac_threshold,
                min_inliers=20
            )
            
            if R is not None:
                translation_mag = np.linalg.norm(t)
                print(f"    Inliers: {num_inliers}/{len(matches)}")
                print(f"    Translation: {translation_mag:.3f} m")
                
                # Add to trajectory
                trajectory.add_pose(R, t)
                
                # Visualize matches (save first 3 and every 10th)
                if i < 3 or i % 10 == 0:
                    match_vis_path = os.path.join(vis_dir, f"matches_{i:06d}_{i+1:06d}.png")
                    plot_feature_matches(
                        left_i, left_i1,
                        kp1, kp2, matches,
                        inliers=inliers,
                        max_display=100,
                        title=f"Matches Frame {i}->{i+1} (Inliers: {num_inliers})",
                        save_path=match_vis_path,
                        show=False
                    )
            else:
                print(f"    Pose estimation FAILED (insufficient inliers: {num_inliers})")
                trajectory.add_pose(None, None)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            trajectory.add_pose(None, None)
            continue
    
    # Get trajectory results
    positions = trajectory.get_positions()
    trajectory_length = trajectory.get_trajectory_length()
    
    print(f"\n{'='*60}")
    print(f"Trajectory Summary:")
    print(f"  Total frames: {trajectory.num_frames}")
    print(f"  Trajectory length: {trajectory_length:.2f} meters")
    print(f"  Final position: {positions[-1]}")
    print(f"{'='*60}")
    
    # Save trajectory
    trajectory_path = os.path.join(trajectory_dir, f"{sequence_id:02d}.txt")
    trajectory.save(trajectory_path)
    
    # Visualize trajectory
    print(f"\nCreating trajectory visualizations...")
    
    # 2D plot
    traj_2d_path = os.path.join(vis_dir, f"trajectory_2d_seq_{sequence_id:02d}.png")
    plot_trajectory_2d(
        positions,
        title=f"Trajectory - Sequence {sequence_id:02d} (Top View)",
        save_path=traj_2d_path,
        show=False
    )
    
    # 3D plot
    traj_3d_path = os.path.join(vis_dir, f"trajectory_3d_seq_{sequence_id:02d}.png")
    plot_trajectory_3d(
        positions,
        title=f"Trajectory - Sequence {sequence_id:02d} (3D View)",
        save_path=traj_3d_path,
        show=False
    )
    
    print(f"\n{'='*60}")
    print(f"Visual Odometry complete!")
    print(f"Results saved to:")
    print(f"  Trajectory: {trajectory_path}")
    print(f"  Visualizations: {vis_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run visual odometry on KITTI dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--sequence', type=int, default=0,
                       help='Sequence ID to process')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to process')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    dataset_path = config['dataset']['kitti_odometry_path']
    
    print(f"\nVisual Odometry")
    print(f"Dataset: {dataset_path}")
    print(f"Sequence: {args.sequence:02d}")
    print(f"Frames: {args.frames}")
    
    # Run pipeline
    run_vo_pipeline(dataset_path, args.sequence, args.frames, config, args.output)


if __name__ == "__main__":
    main()