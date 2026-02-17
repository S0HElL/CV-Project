"""
Pose estimation with metric scale recovery using stereo depth.
Implements PnP (Perspective-n-Point) with RANSAC for scale recovery.
"""

import numpy as np
import cv2


def triangulate_points_stereo(points_left, disparity_map, focal_length, baseline, cx, cy):
    """
    Convert 2D matched points to 3D using stereo disparity/depth.
    
    Args:
        points_left: Nx2 array of (x, y) coordinates in left image
        disparity_map: Disparity map from stereo matching
        focal_length: Camera focal length (pixels)
        baseline: Stereo baseline (meters)
        cx: Principal point x-coordinate
        cy: Principal point y-coordinate
        
    Returns:
        points_3d: Nx3 array of 3D points in camera coordinates
        valid_mask: Boolean mask indicating which points have valid depth
    """
    num_points = len(points_left)
    points_3d = np.zeros((num_points, 3), dtype=np.float32)
    valid_mask = np.zeros(num_points, dtype=bool)
    
    for i, (x, y) in enumerate(points_left):
        # Convert to integer coordinates for disparity lookup
        x_int = int(round(x))
        y_int = int(round(y))
        
        # Check bounds
        if y_int < 0 or y_int >= disparity_map.shape[0] or x_int < 0 or x_int >= disparity_map.shape[1]:
            continue
        
        # Get disparity at this point
        d = disparity_map[y_int, x_int]
        
        # Skip if disparity is invalid
        if d <= 0:
            continue
        
        # Compute depth: Z = f * B / d
        Z = focal_length * baseline / d
        
        # Skip if depth is unreasonable
        if Z <= 0 or Z > 100:  # Limit to 100 meters
            continue
        
        # Compute X and Y in 3D camera coordinates
        X = (x - cx) * Z / focal_length
        Y = (y - cy) * Z / focal_length
        
        points_3d[i] = [X, Y, Z]
        valid_mask[i] = True
    
    return points_3d, valid_mask


def estimate_pose_pnp(points_3d, points_2d, K, method=cv2.SOLVEPNP_ITERATIVE,
                      ransac=True, ransac_threshold=8.0, ransac_iterations=1000,
                      ransac_confidence=0.99):
    """
    Estimate camera pose using PnP (Perspective-n-Point).
    
    Solves for camera pose given 3D-2D correspondences:
    - 3D points in frame t (camera coordinates)
    - 2D points in frame t+1 (image coordinates)
    
    Args:
        points_3d: Nx3 array of 3D points
        points_2d: Nx2 array of corresponding 2D points
        K: 3x3 camera intrinsic matrix
        method: PnP method (ITERATIVE, P3P, EPNP, etc.)
        ransac: Whether to use RANSAC
        ransac_threshold: RANSAC reprojection error threshold (pixels)
        ransac_iterations: Maximum RANSAC iterations
        ransac_confidence: RANSAC confidence level
        
    Returns:
        R: 3x3 rotation matrix (camera t to camera t+1)
        t: 3x1 translation vector
        inlier_mask: Boolean mask of inliers
        success: Boolean indicating if estimation succeeded
    """
    if len(points_3d) < 4:
        return None, None, np.zeros(len(points_3d), dtype=bool), False
    
    # Ensure correct data types
    points_3d = points_3d.astype(np.float32)
    points_2d = points_2d.astype(np.float32)
    
    # No distortion coefficients (KITTI is rectified)
    dist_coeffs = np.zeros(4)
    
    if ransac:
        # Use solvePnPRansac for robust estimation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            dist_coeffs,
            iterationsCount=ransac_iterations,
            reprojectionError=ransac_threshold,
            confidence=ransac_confidence,
            flags=method
        )
        
        # Convert inliers to boolean mask
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        if inliers is not None:
            inlier_mask[inliers.flatten()] = True
    else:
        # Use standard solvePnP
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            K,
            dist_coeffs,
            flags=method
        )
        inlier_mask = np.ones(len(points_3d), dtype=bool)
    
    if not success:
        return None, None, inlier_mask, False
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec
    
    return R, t, inlier_mask, True


def estimate_pose_with_scale(points1, points2, disparity_map, K, baseline,
                             ransac_threshold=8.0, min_inliers=20):
    """
    Estimate camera pose with metric scale using stereo depth.
    
    Pipeline:
    1. Convert matched 2D points from frame t to 3D using stereo depth
    2. Use PnP with RANSAC to estimate pose to frame t+1
    3. Returns metric-scale transformation
    
    Args:
        points1: Nx2 array of points in frame t (left image)
        points2: Nx2 array of points in frame t+1 (left image)
        disparity_map: Disparity map from frame t
        K: 3x3 camera intrinsic matrix
        baseline: Stereo baseline (meters)
        ransac_threshold: RANSAC reprojection threshold (pixels)
        min_inliers: Minimum number of inliers required
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector (with metric scale)
        inlier_mask: Boolean mask of inliers
        num_inliers: Number of inliers
    """
    # Extract calibration parameters
    focal_length = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Convert 2D points from frame t to 3D using stereo depth
    points_3d, valid_3d_mask = triangulate_points_stereo(
        points1, disparity_map, focal_length, baseline, cx, cy
    )
    
    # Filter out points without valid depth
    if np.sum(valid_3d_mask) < 4:
        return None, None, np.zeros(len(points1), dtype=bool), 0
    
    valid_points_3d = points_3d[valid_3d_mask]
    valid_points_2d = points2[valid_3d_mask]
    
    # Estimate pose using PnP with RANSAC
    R, t, pnp_inlier_mask, success = estimate_pose_pnp(
        valid_points_3d,
        valid_points_2d,
        K,
        ransac=True,
        ransac_threshold=ransac_threshold,
        ransac_iterations=1000,
        ransac_confidence=0.99
    )
    
    if not success:
        return None, None, np.zeros(len(points1), dtype=bool), 0
    
    # Map PnP inliers back to original point indices
    full_inlier_mask = np.zeros(len(points1), dtype=bool)
    valid_indices = np.where(valid_3d_mask)[0]
    pnp_inlier_indices = valid_indices[pnp_inlier_mask]
    full_inlier_mask[pnp_inlier_indices] = True
    
    num_inliers = np.sum(full_inlier_mask)
    
    # Check if we have enough inliers
    if num_inliers < min_inliers:
        return None, None, full_inlier_mask, num_inliers
    
    return R, t, full_inlier_mask, num_inliers


def invert_transformation(R, t):
    """
    Invert a rigid transformation (R, t).
    
    If [R|t] transforms points from A to B,
    then [R^T | -R^T * t] transforms from B to A.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        R_inv: Inverted rotation matrix
        t_inv: Inverted translation vector
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def compose_transformations(R1, t1, R2, t2):
    """
    Compose two transformations: T = T2 * T1
    
    Args:
        R1, t1: First transformation
        R2, t2: Second transformation
        
    Returns:
        R: Composed rotation
        t: Composed translation
    """
    R = R2 @ R1
    t = R2 @ t1 + t2
    return R, t


def transformation_to_matrix(R, t):
    """
    Convert (R, t) to 4x4 homogeneous transformation matrix.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        T: 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def matrix_to_transformation(T):
    """
    Extract (R, t) from 4x4 homogeneous transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    return R, t


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    from src.utils.kitti_odometry_loader import load_left_image, load_calibration
    from src.vo.features import detect_and_match
    from src.stereo.block_matching import compute_disparity_optimized
    
    print("Testing pose estimation with metric scale")
    print("=" * 60)
    
    # Test configuration
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    frame_id = 0
    
    print(f"\nTesting on sequence {sequence_id:02d}, frames {frame_id} and {frame_id+1}")
    
    try:
        # Load calibration
        print(f"\n1. Loading calibration")
        calib = load_calibration(dataset_path, sequence_id)
        K = calib['K']
        focal_length = calib['f']
        baseline = calib['B']
        print(f"   Focal length: {focal_length:.2f} pixels")
        print(f"   Baseline: {baseline:.4f} meters")
        
        # Load stereo pair for frame t
        print(f"\n2. Loading stereo pair for frame {frame_id}")
        left_t, right_t = load_stereo_pair(dataset_path, sequence_id, frame_id)
        print(f"   Image shape: {left_t.shape}")
        
        # Compute disparity for frame t
        print(f"\n3. Computing disparity map")
        disparity = compute_disparity_optimized(
            left_t, right_t,
            window_size=11,
            max_disparity=128,
            cost_function='SAD'
        )
        valid_disp = np.sum(disparity > 0)
        print(f"   Valid disparity pixels: {valid_disp} ({100*valid_disp/disparity.size:.1f}%)")
        
        # Load frame t+1
        print(f"\n4. Loading frame {frame_id+1}")
        left_t1 = load_left_image(dataset_path, sequence_id, frame_id + 1)
        
        # Detect and match features between frames
        print(f"\n5. Detecting and matching features")
        kp1, kp2, matches, points1, points2 = detect_and_match(
            left_t, left_t1,
            detector_type='ORB',
            max_features=3000,
            ratio_threshold=0.75
        )
        print(f"   Feature matches: {len(matches)}")
        
        # Test 3D point triangulation
        print(f"\n6. Triangulating 3D points from stereo")
        points_3d, valid_3d = triangulate_points_stereo(
            points1, disparity,
            focal_length, baseline,
            K[0, 2], K[1, 2]
        )
        num_valid_3d = np.sum(valid_3d)
        print(f"   Valid 3D points: {num_valid_3d}/{len(points1)}")
        
        if num_valid_3d > 0:
            valid_depths = points_3d[valid_3d, 2]
            print(f"   Depth range: [{valid_depths.min():.2f}, {valid_depths.max():.2f}] meters")
            print(f"   Mean depth: {valid_depths.mean():.2f} meters")
        
        # Test PnP estimation
        if num_valid_3d >= 4:
            print(f"\n7. Estimating pose using PnP + RANSAC")
            R_pnp, t_pnp, pnp_inliers, success = estimate_pose_pnp(
                points_3d[valid_3d],
                points2[valid_3d],
                K,
                ransac=True,
                ransac_threshold=8.0
            )
            
            if success:
                print(f"   PnP estimation successful")
                print(f"   Inliers: {np.sum(pnp_inliers)}/{num_valid_3d}")
                print(f"   Rotation matrix R:")
                print(f"   {R_pnp}")
                print(f"   Translation t (meters):")
                print(f"   {t_pnp.flatten()}")
                print(f"   Translation magnitude: {np.linalg.norm(t_pnp):.3f} meters")
            else:
                print(f"   PnP estimation failed")
        
        # Test complete pipeline
        print(f"\n8. Testing complete pose estimation pipeline")
        R, t, inliers, num_inliers = estimate_pose_with_scale(
            points1, points2,
            disparity,
            K, baseline,
            ransac_threshold=8.0,
            min_inliers=20
        )
        
        if R is not None:
            print(f"   Pose estimation successful")
            print(f"   Total matches: {len(matches)}")
            print(f"   Inliers: {num_inliers}")
            print(f"   Inlier ratio: {100*num_inliers/len(matches):.1f}%")
            print(f"   Translation (meters): {t.flatten()}")
            print(f"   Translation magnitude: {np.linalg.norm(t):.3f} meters")
            
            # Compute rotation angle
            angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
            print(f"   Rotation angle: {angle:.2f} degrees")
        else:
            print(f"   Pose estimation failed (insufficient inliers: {num_inliers})")
        
        # Test transformation utilities
        print(f"\n9. Testing transformation utilities")
        if R is not None:
            T = transformation_to_matrix(R, t)
            print(f"   Transformation matrix T:")
            print(f"   {T}")
            
            R_back, t_back = matrix_to_transformation(T)
            print(f"   Conversion check: {np.allclose(R, R_back) and np.allclose(t, t_back)}")
            
            R_inv, t_inv = invert_transformation(R, t)
            print(f"   Inverted translation: {t_inv.flatten()}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Pose estimation with metric scale is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()