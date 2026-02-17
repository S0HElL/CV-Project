#!/usr/bin/env python3
"""
Script to evaluate visual odometry trajectory against ground truth.
"""

import sys
import os
import yaml
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.kitti_odometry_loader import load_ground_truth_poses
from src.vo.trajectory import load_trajectory, extract_trajectory_positions
from src.evaluation.vo_metrics import compute_vo_metrics_summary, print_vo_metrics
from src.visualization.plot_trajectory import plot_trajectory_comparison, plot_trajectory_error


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_vo(trajectory_path, gt_poses, sequence_id, output_dir):
    """
    Evaluate visual odometry trajectory.
    
    Args:
        trajectory_path: Path to estimated trajectory file
        gt_poses: List of ground truth pose matrices
        sequence_id: Sequence ID
        output_dir: Output directory for visualizations
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Trajectory: Sequence {sequence_id:02d}")
    print(f"{'='*60}")
    
    # Load estimated trajectory
    print(f"\nLoading estimated trajectory from {trajectory_path}")
    poses_est = load_trajectory(trajectory_path)
    print(f"  Estimated poses: {len(poses_est)}")
    print(f"  Ground truth poses: {len(gt_poses)}")
    
    # Align lengths
    min_len = min(len(poses_est), len(gt_poses))
    poses_est = poses_est[:min_len]
    poses_gt = gt_poses[:min_len]
    
    print(f"  Evaluating {min_len} frames")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_vo_metrics_summary(poses_est, poses_gt, rpe_steps=[1, 5, 10])
    
    # Print results
    print_vo_metrics(metrics, f"VO Evaluation - Sequence {sequence_id:02d}")
    
    # Extract positions
    positions_est = extract_trajectory_positions(poses_est)
    positions_gt = extract_trajectory_positions(poses_gt)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    vis_dir = os.path.join(output_dir, f"eval_seq_{sequence_id:02d}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Trajectory comparison
    comparison_path = os.path.join(vis_dir, f"trajectory_comparison_seq_{sequence_id:02d}.png")
    plot_trajectory_comparison(
        positions_est, positions_gt,
        title=f"Trajectory Comparison - Sequence {sequence_id:02d}",
        save_path=comparison_path,
        show=False
    )
    
    # Error plot
    error_path = os.path.join(vis_dir, f"trajectory_error_seq_{sequence_id:02d}.png")
    plot_trajectory_error(
        positions_est, positions_gt,
        title=f"Trajectory Error - Sequence {sequence_id:02d}",
        save_path=error_path,
        show=False
    )
    
    print(f"\nVisualizations saved to {vis_dir}")
    
    # Save metrics to file
    metrics_path = os.path.join(vis_dir, f"metrics_seq_{sequence_id:02d}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Visual Odometry Evaluation - Sequence {sequence_id:02d}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Trajectory Info:\n")
        f.write(f"  Frames: {metrics['num_frames']}\n")
        f.write(f"  Estimated length: {metrics['trajectory_length_est']:.2f} m\n")
        f.write(f"  Ground truth length: {metrics['trajectory_length_gt']:.2f} m\n\n")
        f.write(f"ATE (Absolute Trajectory Error):\n")
        f.write(f"  RMSE: {metrics['ate']['mean']:.4f} m\n")
        f.write(f"  Std: {metrics['ate']['std']:.4f} m\n")
        f.write(f"  Median: {metrics['ate']['median']:.4f} m\n")
        f.write(f"  Min: {metrics['ate']['min']:.4f} m\n")
        f.write(f"  Max: {metrics['ate']['max']:.4f} m\n\n")
        f.write(f"RPE (Relative Pose Error):\n")
        for step_name, rpe in metrics['rpe'].items():
            step = int(step_name.split('_')[1])
            f.write(f"  Step {step}:\n")
            f.write(f"    Translation RMSE: {rpe['trans_mean']:.4f} m\n")
            f.write(f"    Rotation RMSE: {rpe['rot_mean']:.4f} deg\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate visual odometry')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--trajectory', type=str, required=True,
                       help='Path to estimated trajectory file')
    parser.add_argument('--sequence', type=int, required=True,
                       help='Sequence ID')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    dataset_path = config['dataset']['kitti_odometry_path']
    
    print(f"\nVisual Odometry Evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Sequence: {args.sequence:02d}")
    print(f"Trajectory: {args.trajectory}")
    
    # Load ground truth
    print(f"\nLoading ground truth poses...")
    gt_poses = load_ground_truth_poses(dataset_path, args.sequence)
    
    if gt_poses is None:
        print(f"\nERROR: Ground truth poses not available for sequence {args.sequence:02d}")
        print(f"Please download ground truth poses from KITTI odometry benchmark")
        print(f"and place them in: {dataset_path}/poses/{args.sequence:02d}.txt")
        return
    
    # Evaluate
    evaluate_vo(args.trajectory, gt_poses, args.sequence, args.output)


if __name__ == "__main__":
    main()