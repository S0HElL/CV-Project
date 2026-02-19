#!/usr/bin/env python3
"""
Script to compute stereo disparity and depth maps.
Processes KITTI sequences and saves results.
"""

import sys
import os
import yaml
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.kitti_stereo_loader import load_stereo_pair, get_sequence_length
from src.utils.kitti_odometry_loader import load_calibration
from src.stereo.block_matching import compute_disparity_optimized
from src.stereo.sgbm import compute_disparity_sgbm
from src.stereo.consistency import compute_lr_consistency, compute_lr_consistency_relaxed
from src.stereo.postprocessing import postprocess_disparity
from src.stereo.depth import disparity_to_depth
from src.visualization.plot_disparity import plot_disparity_map, plot_disparity_with_image
from src.visualization.plot_depth import plot_depth_map, plot_depth_with_image


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_stereo_pipeline(dataset_path, sequence_id, frame_ids, config, output_dir):
    """
    Run stereo pipeline on specified frames.
    
    Args:
        dataset_path: Path to KITTI dataset
        sequence_id: Sequence number
        frame_ids: List of frame IDs to process
        config: Configuration dictionary
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Processing Sequence {sequence_id:02d}")
    print(f"{'='*60}")
    
    # Load calibration
    print(f"\nLoading calibration...")
    calib = load_calibration(dataset_path, sequence_id)
    focal_length = calib['f']
    baseline = calib['B']
    print(f"  Focal length: {focal_length:.2f} pixels")
    print(f"  Baseline: {baseline:.4f} meters")
    
    # Stereo parameters
    window_size = config['stereo']['window_size']
    max_disparity = config['stereo']['max_disparity']
    cost_function = config['stereo']['cost_function']
    
    print(f"\nStereo parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Max disparity: {max_disparity}")
    print(f"  Cost function: {cost_function}")
    
    # Create output directories
    method = config['stereo'].get('method', 'sgbm')
    disparity_dir = os.path.join(config['output']['disparity_dir'], f"seq_{sequence_id:02d}_{method}")
    depth_dir = os.path.join(config['output']['depth_dir'], f"seq_{sequence_id:02d}_{method}")
    vis_dir = os.path.join(config['output']['visualizations_dir'], f"seq_{sequence_id:02d}_{method}")

    os.makedirs(disparity_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each frame
    for frame_id in frame_ids:
        print(f"\n--- Frame {frame_id:06d} ---")
        
        try:
            # Load stereo pair
            left_img, right_img = load_stereo_pair(dataset_path, sequence_id, frame_id)
            print(f"  Loaded images: {left_img.shape}")
            
            # Compute disparity (left-to-right)
            method = config['stereo'].get('method', 'sgbm')

            print(f"  Computing disparity (L->R) using {method}...")

            if method == "sgbm":
                disparity_lr = compute_disparity_sgbm(
                    left_img,
                    right_img,
                    min_disparity=0,
                    num_disparities=max_disparity,
                    block_size=window_size
                )

            elif method == "block_matching":
                disparity_lr = compute_disparity_optimized(
                    left_img,
                    right_img,
                    window_size=window_size,
                    max_disparity=max_disparity,
                    cost_function=cost_function
                )

            else:
                raise ValueError(f"Unknown stereo method: {method}")

            # Conditionally compute R->L and consistency check
            if method == "block_matching":
                # Block matching benefits from consistency check
                print(f"  Computing disparity (R->L)...")
                disparity_rl = compute_disparity_optimized(
                    right_img, left_img,
                    window_size=window_size,
                    max_disparity=max_disparity,
                    cost_function=cost_function
                )
                
                # Consistency check with relaxed thresholds
                print(f"  Applying consistency check...")
                consistent_disparity, valid_mask = compute_lr_consistency_relaxed(
                    disparity_lr, disparity_rl,
                    threshold=3.0,      # Good matches
                    max_threshold=10.0  # Acceptable matches
                )
                
                num_valid_before = np.sum(disparity_lr > 0)
                num_valid_after = np.sum(consistent_disparity > 0)
                print(f"    Valid pixels: {num_valid_before} -> {num_valid_after}")

            elif method == "sgbm":
                # SGBM already has internal consistency - skip check
                print(f"  Skipping consistency check (SGBM has internal validation)")
                consistent_disparity = disparity_lr
                valid_mask = disparity_lr > 0
                num_valid_before = np.sum(disparity_lr > 0)
                print(f"    Valid pixels: {num_valid_before}")

            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Post-processing
            print(f"  Post-processing disparity...")
            processed_disparity = postprocess_disparity(
                consistent_disparity,
                valid_mask=valid_mask,
                median_kernel=config['stereo']['median_filter_size'],
                fill_method='horizontal',
                remove_speckles=True,
                min_region_size=20
            )
            
            num_valid_final = np.sum(processed_disparity > 0)
            print(f"    Final valid pixels: {num_valid_final}")
            
            # Convert to depth
            print(f"  Converting to depth...")
            depth_map = disparity_to_depth(processed_disparity, focal_length, baseline)
            
            valid_depth = depth_map[depth_map > 0]
            if len(valid_depth) > 0:
                print(f"    Depth range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] meters")
                print(f"    Mean depth: {valid_depth.mean():.2f} meters")
            
            # Save disparity
            disparity_path = os.path.join(disparity_dir, f"{frame_id:06d}.npy")
            np.save(disparity_path, processed_disparity)
            print(f"  Saved disparity to {disparity_path}")
            
            # Save depth
            depth_path = os.path.join(depth_dir, f"{frame_id:06d}.npy")
            np.save(depth_path, depth_map)
            print(f"  Saved depth to {depth_path}")
            
            # Visualize
            print(f"  Creating visualizations...")
            
            # Disparity visualization
            disp_vis_path = os.path.join(vis_dir, f"disparity_{frame_id:06d}.png")
            plot_disparity_with_image(
                left_img, processed_disparity,
                title=f"Disparity - Seq {sequence_id:02d} Frame {frame_id:06d}",
                save_path=disp_vis_path,
                show=False
            )
            
            # Depth visualization
            depth_vis_path = os.path.join(vis_dir, f"depth_{frame_id:06d}.png")
            plot_depth_with_image(
                left_img, depth_map,
                title=f"Depth - Seq {sequence_id:02d} Frame {frame_id:06d}",
                save_path=depth_vis_path,
                show=False
            )
            
        except Exception as e:
            print(f"  ERROR processing frame {frame_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Stereo processing complete!")
    print(f"Results saved to:")
    print(f"  Disparity: {disparity_dir}")
    print(f"  Depth: {depth_dir}")
    print(f"  Visualizations: {vis_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run stereo depth estimation on KITTI dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--sequence', type=int, default=0,
                       help='Sequence ID to process')
    parser.add_argument('--frames', type=str, default='0-10',
                       help='Frame range (e.g., "0-10" or "0,5,10")')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--method', type=str, choices=['sgbm', 'block_matching'],
                       help='Override stereo method')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override method if specified in command line
    if args.method:
        config['stereo']['method'] = args.method
    
    dataset_path = config['dataset']['kitti_odometry_path']
    
    # Parse frame range
    if '-' in args.frames:
        start, end = map(int, args.frames.split('-'))
        frame_ids = list(range(start, end + 1))
    else:
        frame_ids = [int(x) for x in args.frames.split(',')]
    
    print(f"\nStereo Depth Estimation")
    print(f"Dataset: {dataset_path}")
    print(f"Sequence: {args.sequence:02d}")
    print(f"Frames: {frame_ids}")
    
    # Run pipeline
    run_stereo_pipeline(dataset_path, args.sequence, frame_ids, config, args.output)


if __name__ == "__main__":
    main()