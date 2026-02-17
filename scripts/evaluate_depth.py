#!/usr/bin/env python3
"""
Script to evaluate depth estimation against KITTI ground truth.
Note: KITTI odometry does not include disparity ground truth.
This script is a template for evaluation if ground truth is available.
"""

import sys
import os
import yaml
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.depth_metrics import compute_depth_metrics_summary, print_metrics


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_depth(disparity_pred_path, disparity_gt_path, focal_length, baseline):
    """
    Evaluate depth estimation.
    
    Args:
        disparity_pred_path: Path to predicted disparity (.npy)
        disparity_gt_path: Path to ground truth disparity (.npy)
        focal_length: Camera focal length
        baseline: Stereo baseline
    """
    # Load disparities
    disparity_pred = np.load(disparity_pred_path)
    disparity_gt = np.load(disparity_gt_path)
    
    print(f"\nEvaluating: {os.path.basename(disparity_pred_path)}")
    print(f"  Predicted shape: {disparity_pred.shape}")
    print(f"  Ground truth shape: {disparity_gt.shape}")
    
    # Compute metrics
    metrics = compute_depth_metrics_summary(
        disparity_pred, disparity_gt,
        focal_length=focal_length,
        baseline=baseline,
        bad_pixel_threshold=3.0
    )
    
    # Print results
    print_metrics(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth estimation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted disparities')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth disparities')
    parser.add_argument('--sequence', type=int, default=0,
                       help='Sequence ID')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"\nDepth Evaluation")
    print(f"{'='*60}")
    print(f"Predicted: {args.pred_dir}")
    print(f"Ground truth: {args.gt_dir}")
    print(f"Sequence: {args.sequence:02d}")
    
    # Note: KITTI odometry dataset does not include disparity ground truth
    # This is a template for evaluation when ground truth is available
    # (e.g., KITTI Stereo 2015 dataset)
    
    print(f"\nNOTE: KITTI Odometry dataset does not include disparity ground truth.")
    print(f"For proper depth evaluation, use KITTI Stereo 2015 dataset.")
    print(f"This script serves as a template for when ground truth is available.")
    
    # Example usage (commented out):
    # from src.utils.kitti_odometry_loader import load_calibration
    # calib = load_calibration(config['dataset']['kitti_odometry_path'], args.sequence)
    # 
    # # Evaluate all files in directory
    # pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.npy')])
    # 
    # all_metrics = []
    # for pred_file in pred_files:
    #     pred_path = os.path.join(args.pred_dir, pred_file)
    #     gt_path = os.path.join(args.gt_dir, pred_file)
    #     
    #     if os.path.exists(gt_path):
    #         metrics = evaluate_depth(pred_path, gt_path, calib['f'], calib['B'])
    #         all_metrics.append(metrics)
    # 
    # # Compute average metrics
    # if len(all_metrics) > 0:
    #     avg_mae = np.mean([m['disparity_mae'] for m in all_metrics])
    #     avg_bad_pixel = np.mean([m['bad_pixel_rate'] for m in all_metrics])
    #     print(f"\nAverage Metrics:")
    #     print(f"  MAE: {avg_mae:.3f} pixels")
    #     print(f"  Bad-pixel rate: {avg_bad_pixel:.2f}%")


if __name__ == "__main__":
    main()