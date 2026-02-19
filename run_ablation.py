#!/usr/bin/env python3
"""
Ablation Study Script for Visual Odometry.
Evaluates:
1. Impact of RANSAC (Geometric robustness)
2. Impact of Scale (Stereo/GT scale vs Monocular scale)
"""

import sys
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. SETUP PATHS ---
# Add project root to sys.path so we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- 2. IMPORTS ---
# Import using your project structure
from src.vo import geometry as geo
from src.utils.kitti_odometry_loader import load_left_image, load_calibration, load_ground_truth_poses
from src.vo.features import detect_and_match

def get_ground_truth_scale(gt_poses, frame_id):
    """Calculates the scalar distance moved between frame-1 and frame based on GT."""
    if frame_id == 0: return 0.0
    
    pose_prev = gt_poses[frame_id - 1]
    pose_curr = gt_poses[frame_id]
    
    pos_prev = pose_prev[:3, 3]
    pos_curr = pose_curr[:3, 3]
    
    return np.linalg.norm(pos_curr - pos_prev)

def form_transform(R, t, scale=1.0):
    """Forms a 4x4 homogenous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten() * scale
    return T

def run_ablation(dataset_path, sequence_id, max_frames=None):
    print(f"\n{'='*60}")
    print(f"Running Ablation Study: Sequence {sequence_id:02d}")
    print(f"{'='*60}\n")
    
    # Load Calibration and Ground Truth
    print("Loading data...")
    calib = load_calibration(dataset_path, sequence_id)
    K = calib['K']
    gt_poses = load_ground_truth_poses(dataset_path, sequence_id)
    
    if max_frames is None:
        max_frames = len(gt_poses)
    else:
        max_frames = min(max_frames, len(gt_poses))
    
    # Initialize Trajectories (Global Poses)
    # 1. Baseline: RANSAC + GT Scale (Represents Stereo/Ideal VO)
    # 2. No RANSAC: High Threshold + GT Scale (Shows effect of outliers)
    # 3. Mono Scale: RANSAC + Unit Scale (Shows scale drift)
    
    trajectories = {
        'baseline': [np.eye(4)],
        'no_ransac': [np.eye(4)],
        'mono': [np.eye(4)]
    }
    
    # Current global pose accumulators
    curr_pose = {
        'baseline': np.eye(4),
        'no_ransac': np.eye(4),
        'mono': np.eye(4)
    }

    img_prev = load_left_image(dataset_path, sequence_id, 0)
    
    print(f"Processing {max_frames} frames...")
    
    for i in tqdm(range(1, max_frames)):
        img_curr = load_left_image(dataset_path, sequence_id, i)
        
        # --- 1. Feature Matching (Shared) ---
        kp1, kp2, matches, pts1, pts2 = detect_and_match(
            img_prev, img_curr,
            detector_type='ORB', # Using ORB as per your previous setup
            max_features=3000,
            ratio_threshold=0.75
        )
        
        if len(matches) < 8:
            for k in trajectories: trajectories[k].append(curr_pose[k])
            continue

        # --- 2. ESTIMATE ESSENTIAL MATRIX (Ablation: RANSAC) ---
        
        # A. With RANSAC (Strict threshold = 1.0)
        E_ransac, mask_ransac = geo.estimate_essential_matrix(
            pts1, pts2, K, 
            method=cv2.RANSAC, 
            threshold=1.0, 
            confidence=0.99
        )
        
        # B. Without RANSAC (Loose threshold = 50.0)
        # We simulate "No RANSAC" by allowing outliers to pass
        E_no_ransac, mask_no_ransac = geo.estimate_essential_matrix(
            pts1, pts2, K, 
            method=cv2.RANSAC, 
            threshold=50.0, 
            confidence=0.99
        )

        # --- 3. RECOVER POSE ---
        
        # Recover pose for Baseline & Mono (using good mask)
        R_good, t_good, _ = geo.recover_pose_from_essential(E_ransac, pts1, pts2, K, mask_ransac)
        
        # Recover pose for No-RANSAC (using bad mask)
        R_bad, t_bad, _ = geo.recover_pose_from_essential(E_no_ransac, pts1, pts2, K, mask_no_ransac)
        
        # --- 4. DETERMINE SCALE (Ablation: Scale) ---
        
        gt_scale = get_ground_truth_scale(gt_poses, i)
        mono_scale = 1.0 # Unknown scale

        # --- 5. UPDATE TRAJECTORIES ---
        
        # Scenario 1: Baseline (RANSAC + GT Scale)
        if R_good is not None:
            T_rel = form_transform(R_good, t_good, scale=gt_scale)
            curr_pose['baseline'] = curr_pose['baseline'] @ T_rel
        trajectories['baseline'].append(curr_pose['baseline'])
        
        # Scenario 2: No RANSAC (Bad Geometry + GT Scale)
        if R_bad is not None:
            T_rel = form_transform(R_bad, t_bad, scale=gt_scale)
            curr_pose['no_ransac'] = curr_pose['no_ransac'] @ T_rel
        trajectories['no_ransac'].append(curr_pose['no_ransac'])
        
        # Scenario 3: Monocular (RANSAC + Unit Scale)
        if R_good is not None:
            T_rel = form_transform(R_good, t_good, scale=mono_scale)
            curr_pose['mono'] = curr_pose['mono'] @ T_rel
        trajectories['mono'].append(curr_pose['mono'])

        img_prev = img_curr

    return trajectories, gt_poses[:max_frames]

def plot_results(trajectories, gt_poses, output_dir, seq_id):
    """Generates comparison plots."""
    
    def get_xz(poses):
        return [p[0, 3] for p in poses], [p[2, 3] for p in poses]

    gt_x, gt_z = get_xz(gt_poses)
    base_x, base_z = get_xz(trajectories['baseline'])
    no_ransac_x, no_ransac_z = get_xz(trajectories['no_ransac'])
    mono_x, mono_z = get_xz(trajectories['mono'])

    plt.figure(figsize=(16, 7))

    # Plot 1: RANSAC vs No RANSAC
    plt.subplot(1, 2, 1)
    plt.plot(gt_x, gt_z, 'k--', label='Ground Truth', linewidth=2)
    plt.plot(base_x, base_z, 'g-', label='With RANSAC (Baseline)', linewidth=1.5)
    plt.plot(no_ransac_x, no_ransac_z, 'r-', label='No RANSAC (Noisy)', linewidth=1, alpha=0.7)
    plt.title('Ablation 1: Effect of Geometric Verification (RANSAC)')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')

    # Plot 2: Stereo/GT Scale vs Mono Scale
    plt.subplot(1, 2, 2)
    plt.plot(gt_x, gt_z, 'k--', label='Ground Truth', linewidth=2)
    plt.plot(base_x, base_z, 'g-', label='Stereo/GT Scale', linewidth=1.5)
    plt.plot(mono_x, mono_z, 'b-', label='Monocular Scale (1.0)', linewidth=1)
    plt.title('Ablation 2: Effect of Scale Information')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ablation_study_seq{seq_id:02d}.png")
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    # plt.show() # Uncomment to see window

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run VO Ablation Study')
    parser.add_argument('--dataset', type=str, required=True, help='Path to KITTI dataset root')
    parser.add_argument('--sequence', type=int, default=0, help='Sequence ID')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to process')
    parser.add_argument('--output', type=str, default='outputs/ablation', help='Output directory')
    
    args = parser.parse_args()

    trajectories, gt_poses = run_ablation(args.dataset, args.sequence, args.frames)
    plot_results(trajectories, gt_poses, args.output, args.sequence)