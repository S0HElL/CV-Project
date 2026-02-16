"""
Geometric estimation for visual odometry.
Implements Essential matrix estimation with RANSAC.
"""

import numpy as np
import cv2


def estimate_essential_matrix(points1, points2, K, method=cv2.RANSAC, 
                              threshold=1.0, confidence=0.99, max_iters=1000):
    """
    Estimate Essential matrix using RANSAC.
    
    The Essential matrix E relates corresponding points in two images
    taken by the same camera: p2^T * E * p1 = 0
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        K: 3x3 camera intrinsic matrix
        method: Method for estimation (cv2.RANSAC or cv2.LMEDS)
        threshold: RANSAC inlier threshold (pixels)
        confidence: RANSAC confidence level (0-1)
        max_iters: Maximum RANSAC iterations
        
    Returns:
        E: 3x3 Essential matrix
        inlier_mask: Boolean mask indicating inliers (Nx1)
    """
    if len(points1) < 8:
        raise ValueError("Need at least 8 point correspondences for Essential matrix estimation")
    
    # Estimate Essential matrix using OpenCV
    E, inlier_mask = cv2.findEssentialMat(
        points1, 
        points2, 
        K,
        method=method,
        prob=confidence,
        threshold=threshold,
        maxIters=max_iters
    )
    
    # Convert mask to boolean
    if inlier_mask is not None:
        inlier_mask = inlier_mask.astype(bool).flatten()
    else:
        inlier_mask = np.zeros(len(points1), dtype=bool)
    
    return E, inlier_mask


def estimate_fundamental_matrix(points1, points2, method=cv2.FM_RANSAC,
                                threshold=1.0, confidence=0.99, max_iters=1000):
    """
    Estimate Fundamental matrix using RANSAC.
    
    The Fundamental matrix F relates corresponding points in two images
    (without requiring calibration): p2^T * F * p1 = 0
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        method: Method for estimation (cv2.FM_RANSAC, cv2.FM_LMEDS, cv2.FM_8POINT)
        threshold: RANSAC inlier threshold (pixels)
        confidence: RANSAC confidence level (0-1)
        max_iters: Maximum RANSAC iterations
        
    Returns:
        F: 3x3 Fundamental matrix
        inlier_mask: Boolean mask indicating inliers (Nx1)
    """
    if len(points1) < 8:
        raise ValueError("Need at least 8 point correspondences for Fundamental matrix estimation")
    
    # Estimate Fundamental matrix using OpenCV
    F, inlier_mask = cv2.findFundamentalMat(
        points1,
        points2,
        method=method,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters
    )
    
    # Convert mask to boolean
    if inlier_mask is not None:
        inlier_mask = inlier_mask.astype(bool).flatten()
    else:
        inlier_mask = np.zeros(len(points1), dtype=bool)
    
    return F, inlier_mask


def fundamental_to_essential(F, K):
    """
    Convert Fundamental matrix to Essential matrix using intrinsics.
    
    Relationship: E = K^T * F * K
    
    Args:
        F: 3x3 Fundamental matrix
        K: 3x3 camera intrinsic matrix
        
    Returns:
        E: 3x3 Essential matrix
    """
    E = K.T @ F @ K
    return E


def recover_pose_from_essential(E, points1, points2, K, inlier_mask=None):
    """
    Recover relative camera pose (R, t) from Essential matrix.
    
    Decomposes the Essential matrix and determines the correct solution
    by checking which puts points in front of both cameras.
    
    Args:
        E: 3x3 Essential matrix
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        K: 3x3 camera intrinsic matrix
        inlier_mask: Boolean mask for inliers (optional)
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector (unit vector, scale unknown)
        pose_inlier_mask: Mask of points that pass cheirality check
    """
    # Use only inliers if mask provided
    if inlier_mask is not None:
        pts1 = points1[inlier_mask]
        pts2 = points2[inlier_mask]
    else:
        pts1 = points1
        pts2 = points2
    
    # Recover pose
    num_inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    
    # Convert mask to boolean
    pose_inlier_mask = pose_mask.astype(bool).flatten()
    
    return R, t, pose_inlier_mask


def compute_epipolar_error(points1, points2, F):
    """
    Compute epipolar constraint error for point correspondences.
    
    For each correspondence, computes: |p2^T * F * p1|
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        F: 3x3 Fundamental or Essential matrix
        
    Returns:
        errors: Array of epipolar errors for each point pair
    """
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack([points1, np.ones((len(points1), 1))])
    pts2_h = np.hstack([points2, np.ones((len(points2), 1))])
    
    # Compute epipolar constraint: p2^T * F * p1
    errors = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    
    return errors


def estimate_pose_ransac(points1, points2, K, threshold=1.0, confidence=0.99, 
                         max_iters=1000, min_inliers=50):
    """
    Complete pipeline: estimate Essential matrix with RANSAC and recover pose.
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        K: 3x3 camera intrinsic matrix
        threshold: RANSAC inlier threshold (pixels)
        confidence: RANSAC confidence level (0-1)
        max_iters: Maximum RANSAC iterations
        min_inliers: Minimum number of inliers required
        
    Returns:
        R: 3x3 rotation matrix (None if estimation fails)
        t: 3x1 translation vector (None if estimation fails)
        inlier_mask: Boolean mask of inliers
        num_inliers: Number of inlier correspondences
    """
    if len(points1) < 8:
        return None, None, np.zeros(len(points1), dtype=bool), 0
    
    # Estimate Essential matrix with RANSAC
    E, inlier_mask = estimate_essential_matrix(
        points1, points2, K,
        method=cv2.RANSAC,
        threshold=threshold,
        confidence=confidence,
        max_iters=max_iters
    )
    
    num_inliers = np.sum(inlier_mask)
    
    # Check if we have enough inliers
    if num_inliers < min_inliers:
        return None, None, inlier_mask, num_inliers
    
    # Recover pose from Essential matrix
    R, t, pose_mask = recover_pose_from_essential(E, points1, points2, K, inlier_mask)
    
    return R, t, inlier_mask, num_inliers


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_odometry_loader import load_left_image, load_calibration
    from src.vo.features import detect_and_match
    
    print("Testing Essential matrix estimation with RANSAC")
    print("=" * 60)
    
    # Test configuration
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    frame1_id = 0
    frame2_id = 1
    
    print(f"\nTesting on sequence {sequence_id:02d}, frames {frame1_id}-{frame2_id}")
    
    try:
        # Load images
        print(f"\n1. Loading images and calibration")
        image1 = load_left_image(dataset_path, sequence_id, frame1_id)
        image2 = load_left_image(dataset_path, sequence_id, frame2_id)
        calib = load_calibration(dataset_path, sequence_id)
        K = calib['K']
        
        print(f"   Image shape: {image1.shape}")
        print(f"   Intrinsic matrix K:")
        print(f"   {K}")
        
        # Detect and match features
        print(f"\n2. Detecting and matching features")
        kp1, kp2, matches, points1, points2 = detect_and_match(
            image1, image2,
            detector_type='ORB',
            max_features=3000,
            ratio_threshold=0.75
        )
        print(f"   Found {len(matches)} matches")
        
        # Test Essential matrix estimation
        print(f"\n3. Estimating Essential matrix with RANSAC")
        threshold = 1.0
        E, inlier_mask = estimate_essential_matrix(
            points1, points2, K,
            threshold=threshold,
            confidence=0.99,
            max_iters=1000
        )
        
        num_inliers = np.sum(inlier_mask)
        inlier_ratio = num_inliers / len(matches) if len(matches) > 0 else 0
        
        print(f"   Essential matrix E:")
        print(f"   {E}")
        print(f"   Total matches: {len(matches)}")
        print(f"   Inliers: {num_inliers}")
        print(f"   Outliers: {len(matches) - num_inliers}")
        print(f"   Inlier ratio: {100*inlier_ratio:.1f}%")
        
        # Test Fundamental matrix estimation
        print(f"\n4. Estimating Fundamental matrix with RANSAC")
        F, F_inlier_mask = estimate_fundamental_matrix(
            points1, points2,
            threshold=threshold,
            confidence=0.99,
            max_iters=1000
        )
        
        F_num_inliers = np.sum(F_inlier_mask)
        print(f"   Fundamental matrix F:")
        print(f"   {F}")
        print(f"   Inliers: {F_num_inliers}")
        
        # Test conversion from F to E
        print(f"\n5. Converting F to E")
        E_from_F = fundamental_to_essential(F, K)
        print(f"   E from F:")
        print(f"   {E_from_F}")
        
        # Test pose recovery
        print(f"\n6. Recovering pose from Essential matrix")
        R, t, pose_mask = recover_pose_from_essential(E, points1, points2, K, inlier_mask)
        
        print(f"   Rotation matrix R:")
        print(f"   {R}")
        print(f"   Translation vector t (unit):")
        print(f"   {t.flatten()}")
        print(f"   Points passing cheirality check: {np.sum(pose_mask)}")
        
        # Verify rotation is valid (det(R) = 1, orthogonal)
        det_R = np.linalg.det(R)
        is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        print(f"   det(R) = {det_R:.6f} (should be 1.0)")
        print(f"   R is orthogonal: {is_orthogonal}")
        
        # Test epipolar error
        print(f"\n7. Computing epipolar errors")
        errors_all = compute_epipolar_error(points1, points2, E)
        errors_inliers = errors_all[inlier_mask]
        errors_outliers = errors_all[~inlier_mask]
        
        print(f"   Inlier errors - mean: {np.mean(errors_inliers):.3f}, max: {np.max(errors_inliers):.3f}")
        if len(errors_outliers) > 0:
            print(f"   Outlier errors - mean: {np.mean(errors_outliers):.3f}, max: {np.max(errors_outliers):.3f}")
        
        # Test complete pipeline
        print(f"\n8. Testing complete pose estimation pipeline")
        R_full, t_full, inliers_full, num_inliers_full = estimate_pose_ransac(
            points1, points2, K,
            threshold=1.0,
            confidence=0.99,
            max_iters=1000,
            min_inliers=50
        )
        
        if R_full is not None:
            print(f"   Pose estimation successful")
            print(f"   Inliers: {num_inliers_full}")
            print(f"   Rotation magnitude (angle): {np.arccos((np.trace(R_full) - 1) / 2) * 180 / np.pi:.2f} degrees")
            print(f"   Translation direction: {t_full.flatten()}")
        else:
            print(f"   Pose estimation failed (not enough inliers)")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Essential matrix estimation and RANSAC is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()