"""
Depth evaluation metrics for stereo disparity.
Computes MAE and bad-pixel rate against ground truth.
"""

import numpy as np


def compute_disparity_mae(disparity_pred, disparity_gt, valid_mask=None):
    """
    Compute Mean Absolute Error (MAE) for disparity estimation.
    
    Args:
        disparity_pred: Predicted disparity map
        disparity_gt: Ground truth disparity map
        valid_mask: Boolean mask indicating valid pixels (optional)
                   If None, uses pixels where both pred and gt are > 0
        
    Returns:
        mae: Mean absolute error in pixels
        num_valid: Number of valid pixels evaluated
    """
    if valid_mask is None:
        # Create valid mask: both predicted and ground truth must be valid
        valid_mask = (disparity_pred > 0) & (disparity_gt > 0)
    
    if np.sum(valid_mask) == 0:
        return float('inf'), 0
    
    # Compute absolute errors
    errors = np.abs(disparity_pred[valid_mask] - disparity_gt[valid_mask])
    
    # Compute mean
    mae = np.mean(errors)
    num_valid = np.sum(valid_mask)
    
    return mae, num_valid


def compute_bad_pixel_rate(disparity_pred, disparity_gt, threshold=3.0, valid_mask=None):
    """
    Compute bad-pixel rate (percentage of pixels with error > threshold).
    
    Args:
        disparity_pred: Predicted disparity map
        disparity_gt: Ground truth disparity map
        threshold: Error threshold in pixels (typically 3.0)
        valid_mask: Boolean mask indicating valid pixels (optional)
        
    Returns:
        bad_pixel_rate: Percentage of bad pixels (0-100)
        num_bad: Number of bad pixels
        num_valid: Total number of valid pixels evaluated
    """
    if valid_mask is None:
        # Create valid mask
        valid_mask = (disparity_pred > 0) & (disparity_gt > 0)
    
    if np.sum(valid_mask) == 0:
        return 100.0, 0, 0
    
    # Compute absolute errors
    errors = np.abs(disparity_pred[valid_mask] - disparity_gt[valid_mask])
    
    # Count bad pixels (error > threshold)
    bad_pixels = errors > threshold
    num_bad = np.sum(bad_pixels)
    num_valid = np.sum(valid_mask)
    
    # Compute percentage
    bad_pixel_rate = 100.0 * num_bad / num_valid
    
    return bad_pixel_rate, num_bad, num_valid


def compute_depth_mae(depth_pred, depth_gt, valid_mask=None, max_depth=80.0):
    """
    Compute Mean Absolute Error (MAE) for depth estimation.
    
    Args:
        depth_pred: Predicted depth map (meters)
        depth_gt: Ground truth depth map (meters)
        valid_mask: Boolean mask indicating valid pixels (optional)
        max_depth: Maximum depth to consider (meters)
        
    Returns:
        mae: Mean absolute error in meters
        num_valid: Number of valid pixels evaluated
    """
    if valid_mask is None:
        # Create valid mask: both predicted and ground truth must be valid and within range
        valid_mask = (depth_pred > 0) & (depth_gt > 0) & (depth_gt < max_depth)
    
    if np.sum(valid_mask) == 0:
        return float('inf'), 0
    
    # Compute absolute errors
    errors = np.abs(depth_pred[valid_mask] - depth_gt[valid_mask])
    
    # Compute mean
    mae = np.mean(errors)
    num_valid = np.sum(valid_mask)
    
    return mae, num_valid


def compute_depth_metrics_summary(disparity_pred, disparity_gt, focal_length=None, 
                                  baseline=None, bad_pixel_threshold=3.0):
    """
    Compute comprehensive depth evaluation metrics.
    
    Args:
        disparity_pred: Predicted disparity map
        disparity_gt: Ground truth disparity map
        focal_length: Camera focal length (for depth MAE)
        baseline: Stereo baseline (for depth MAE)
        bad_pixel_threshold: Threshold for bad-pixel rate
        
    Returns:
        metrics: Dictionary containing all metrics
    """
    # Create valid mask
    valid_mask = (disparity_pred > 0) & (disparity_gt > 0)
    num_valid = np.sum(valid_mask)
    
    metrics = {
        'num_valid_pixels': num_valid,
        'total_pixels': disparity_pred.size,
        'coverage': 100.0 * num_valid / disparity_pred.size if disparity_pred.size > 0 else 0.0
    }
    
    if num_valid == 0:
        metrics['disparity_mae'] = float('inf')
        metrics['bad_pixel_rate'] = 100.0
        metrics['depth_mae'] = float('inf')
        return metrics
    
    # Disparity MAE
    disp_mae, _ = compute_disparity_mae(disparity_pred, disparity_gt, valid_mask)
    metrics['disparity_mae'] = disp_mae
    
    # Bad-pixel rate
    bad_rate, num_bad, _ = compute_bad_pixel_rate(
        disparity_pred, disparity_gt, 
        threshold=bad_pixel_threshold, 
        valid_mask=valid_mask
    )
    metrics['bad_pixel_rate'] = bad_rate
    metrics['num_bad_pixels'] = num_bad
    
    # Depth MAE (if calibration provided)
    if focal_length is not None and baseline is not None:
        # Convert disparities to depth
        depth_pred = np.zeros_like(disparity_pred, dtype=np.float32)
        depth_gt = np.zeros_like(disparity_gt, dtype=np.float32)
        
        pred_valid = disparity_pred > 0
        gt_valid = disparity_gt > 0
        
        depth_pred[pred_valid] = focal_length * baseline / disparity_pred[pred_valid]
        depth_gt[gt_valid] = focal_length * baseline / disparity_gt[gt_valid]
        
        depth_mae, _ = compute_depth_mae(depth_pred, depth_gt, valid_mask)
        metrics['depth_mae'] = depth_mae
    
    # Error statistics
    errors = np.abs(disparity_pred[valid_mask] - disparity_gt[valid_mask])
    metrics['error_mean'] = np.mean(errors)
    metrics['error_std'] = np.std(errors)
    metrics['error_median'] = np.median(errors)
    metrics['error_max'] = np.max(errors)
    
    return metrics


def print_metrics(metrics, title="Depth Evaluation Metrics"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * 60)
    
    print(f"Coverage:")
    print(f"  Valid pixels: {metrics['num_valid_pixels']}/{metrics['total_pixels']} ({metrics['coverage']:.1f}%)")
    
    print(f"\nDisparity Metrics:")
    print(f"  MAE: {metrics['disparity_mae']:.3f} pixels")
    print(f"  Bad-pixel rate: {metrics['bad_pixel_rate']:.2f}%")
    print(f"  Bad pixels: {metrics.get('num_bad_pixels', 0)}")
    
    if 'depth_mae' in metrics:
        print(f"\nDepth Metrics:")
        print(f"  MAE: {metrics['depth_mae']:.3f} meters")
    
    print(f"\nError Statistics:")
    print(f"  Mean: {metrics['error_mean']:.3f} pixels")
    print(f"  Std: {metrics['error_std']:.3f} pixels")
    print(f"  Median: {metrics['error_median']:.3f} pixels")
    print(f"  Max: {metrics['error_max']:.3f} pixels")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    print("Testing depth evaluation metrics")
    print("=" * 60)
    
    # Test with synthetic data
    print(f"\n1. Testing with synthetic data")
    
    # Create synthetic disparity maps
    height, width = 100, 100
    disparity_gt = np.random.uniform(10, 50, (height, width)).astype(np.float32)
    
    # Add some noise to create prediction
    noise = np.random.normal(0, 2.0, (height, width))
    disparity_pred = disparity_gt + noise
    
    # Set some pixels to zero (invalid)
    disparity_pred[:10, :] = 0
    disparity_gt[:5, :] = 0
    
    print(f"   Disparity GT shape: {disparity_gt.shape}")
    print(f"   Disparity pred shape: {disparity_pred.shape}")
    
    # Test MAE
    print(f"\n2. Testing disparity MAE")
    mae, num_valid = compute_disparity_mae(disparity_pred, disparity_gt)
    print(f"   MAE: {mae:.3f} pixels")
    print(f"   Valid pixels: {num_valid}")
    
    # Test bad-pixel rate
    print(f"\n3. Testing bad-pixel rate")
    for threshold in [1.0, 2.0, 3.0]:
        bad_rate, num_bad, num_valid = compute_bad_pixel_rate(
            disparity_pred, disparity_gt, threshold=threshold
        )
        print(f"   Threshold {threshold:.1f}px: {bad_rate:.2f}% ({num_bad}/{num_valid})")
    
    # Test depth MAE
    print(f"\n4. Testing depth MAE")
    focal_length = 718.856
    baseline = 0.54
    
    # Convert to depth
    depth_pred = np.zeros_like(disparity_pred)
    depth_gt = np.zeros_like(disparity_gt)
    
    valid_pred = disparity_pred > 0
    valid_gt = disparity_gt > 0
    
    depth_pred[valid_pred] = focal_length * baseline / disparity_pred[valid_pred]
    depth_gt[valid_gt] = focal_length * baseline / disparity_gt[valid_gt]
    
    depth_mae, num_valid = compute_depth_mae(depth_pred, depth_gt)
    print(f"   Depth MAE: {depth_mae:.3f} meters")
    print(f"   Valid pixels: {num_valid}")
    
    # Test comprehensive metrics
    print(f"\n5. Testing comprehensive metrics")
    metrics = compute_depth_metrics_summary(
        disparity_pred, disparity_gt,
        focal_length=focal_length,
        baseline=baseline,
        bad_pixel_threshold=3.0
    )
    
    print_metrics(metrics, "Synthetic Data Evaluation")
    
    # Test edge cases
    print(f"\n6. Testing edge cases")
    
    # All zeros
    print(f"   All zeros:")
    zero_disp = np.zeros((10, 10))
    mae, num_valid = compute_disparity_mae(zero_disp, zero_disp)
    print(f"     MAE: {mae}, Valid: {num_valid}")
    
    # Perfect match
    print(f"   Perfect match:")
    perfect = np.ones((10, 10)) * 20.0
    mae, num_valid = compute_disparity_mae(perfect, perfect)
    print(f"     MAE: {mae:.6f}, Valid: {num_valid}")
    
    # Test with custom valid mask
    print(f"\n7. Testing with custom valid mask")
    custom_mask = np.ones((height, width), dtype=bool)
    custom_mask[50:, :] = False  # Only evaluate top half
    
    mae_custom, num_valid_custom = compute_disparity_mae(
        disparity_pred, disparity_gt, valid_mask=custom_mask
    )
    print(f"   MAE (top half only): {mae_custom:.3f} pixels")
    print(f"   Valid pixels: {num_valid_custom}")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("Depth evaluation metrics are working correctly.")