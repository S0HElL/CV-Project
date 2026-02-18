#!/usr/bin/env python3
"""
Evaluate depth estimation on KITTI Stereo 2015.
"""

import sys
import os
import yaml
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.kitti_stereo2015_loader import (
    load_stereo_2015_pair, 
    load_disparity_ground_truth,
    get_stereo_2015_calibration
)
from src.stereo.sgbm import compute_disparity_sgbm
from src.stereo.block_matching import compute_disparity_optimized
from src.stereo.postprocessing import postprocess_disparity
from src.evaluation.depth_metrics import compute_depth_metrics_summary, print_metrics


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_on_stereo2015(dataset_path, image_ids, method='sgbm', config=None):
    """
    Evaluate depth estimation on KITTI Stereo 2015.
    
    Args:
        dataset_path: Path to Stereo 2015 training folder
        image_ids: List of image IDs to evaluate
        method: 'sgbm' or 'block_matching'
        config: Configuration dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on KITTI Stereo 2015")
    print(f"Method: {method}")
    print(f"Images: {image_ids}")
    print(f"{'='*60}")
    
    # Get calibration
    calib = get_stereo_2015_calibration()
    focal_length = calib['f']
    baseline = calib['B']
    
    all_metrics = []
    
    for image_id in image_ids:
        print(f"\n--- Image {image_id:06d} ---")
        
        try:
            # Load images
            left_img, right_img = load_stereo_2015_pair(dataset_path, image_id)
            print(f"  Loaded images: {left_img.shape}")
            
            # Load ground truth
            disp_gt = load_disparity_ground_truth(dataset_path, image_id, occluded=False)
            print(f"  Loaded GT disparity: {disp_gt.shape}")
            
            # Compute disparity
            print(f"  Computing disparity...")
            if method == 'sgbm':
                disp_pred = compute_disparity_sgbm(
                    left_img, right_img,
                    min_disparity=0,
                    num_disparities=192,
                    block_size=11
                )
            else:
                disp_pred = compute_disparity_optimized(
                    left_img, right_img,
                    window_size=11,
                    max_disparity=192,
                    cost_function='SAD'
                )
            
            # Post-process
            print(f"  Post-processing...")
            valid_mask = disp_pred > 0
            disp_pred = postprocess_disparity(
                disp_pred, valid_mask,
                median_kernel=5,
                fill_method='horizontal',
                remove_speckles=True,
                min_region_size=20
            )
            
            # Evaluate
            print(f"  Computing metrics...")
            metrics = compute_depth_metrics_summary(
                disp_pred, disp_gt,
                focal_length=focal_length,
                baseline=baseline,
                bad_pixel_threshold=3.0
            )
            
            print_metrics(metrics, f"Image {image_id:06d}")
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Compute average metrics
    if len(all_metrics) > 0:
        print(f"\n{'='*60}")
        print(f"Average Metrics Across {len(all_metrics)} Images")
        print(f"{'='*60}")
        
        avg_mae = np.mean([m['disparity_mae'] for m in all_metrics])
        avg_bad = np.mean([m['bad_pixel_rate'] for m in all_metrics])
        avg_coverage = np.mean([m['coverage'] for m in all_metrics])
        
        print(f"\nDisparity Metrics:")
        print(f"  MAE: {avg_mae:.3f} pixels")
        print(f"  Bad-pixel rate: {avg_bad:.2f}%")
        print(f"  Coverage: {avg_coverage:.1f}%")
        
        if 'depth_mae' in all_metrics[0]:
            avg_depth_mae = np.mean([m['depth_mae'] for m in all_metrics])
            print(f"\nDepth Metrics:")
            print(f"  MAE: {avg_depth_mae:.3f} meters")


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth on KITTI Stereo 2015')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--images', type=str, default='0-9',
                       help='Image range (e.g., "0-9" or "0,5,10")')
    parser.add_argument('--method', type=str, default='sgbm',
                       choices=['sgbm', 'block_matching'])
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    dataset_path = config['dataset']['kitti_stereo_path']
    
    # Parse image range
    if '-' in args.images:
        start, end = map(int, args.images.split('-'))
        image_ids = list(range(start, end + 1))
    else:
        image_ids = [int(x) for x in args.images.split(',')]
    
    # Evaluate
    evaluate_on_stereo2015(dataset_path, image_ids, args.method, config)


if __name__ == "__main__":
    main()