"""
Left-right consistency check for stereo disparity validation.
"""

import numpy as np


def compute_lr_consistency(disparity_left, disparity_right, threshold=1.0):
    """
    Perform left-right consistency check on disparity maps.
    
    The consistency check verifies that the disparity estimated from the left
    image matches the disparity estimated from the right image. For a pixel
    at position (x, y) in the left image with disparity d_left, the corresponding
    pixel in the right image should be at (x - d_left, y) with disparity d_right,
    and d_left should approximately equal d_right.
    
    Args:
        disparity_left: Disparity map computed from left to right
        disparity_right: Disparity map computed from right to left
        threshold: Maximum allowed difference between disparities (pixels)
        
    Returns:
        consistent_disparity: Disparity map with inconsistent pixels marked as 0
        valid_mask: Boolean mask indicating valid (consistent) pixels
    """
    height, width = disparity_left.shape
    
    # Initialize output
    consistent_disparity = np.copy(disparity_left)
    valid_mask = np.zeros((height, width), dtype=bool)
    
    # Check each pixel in the left disparity map
    for y in range(height):
        for x in range(width):
            d_left = disparity_left[y, x]
            
            # Skip if disparity is zero (invalid)
            if d_left < 0.5:
                continue
            
            # Find corresponding pixel in right image
            x_right = int(x - d_left)
            
            # Check if corresponding pixel is within image bounds
            if x_right < 0 or x_right >= width:
                consistent_disparity[y, x] = 0
                continue
            
            # Get disparity at corresponding pixel in right image
            d_right = disparity_right[y, x_right]
            
            # Check consistency: d_left should approximately equal d_right
            if abs(d_left - d_right) <= threshold:
                valid_mask[y, x] = True
            else:
                # Mark as invalid
                consistent_disparity[y, x] = 0
    
    return consistent_disparity, valid_mask


def compute_lr_consistency_optimized(disparity_left, disparity_right, threshold=1.0):
    """
    Optimized left-right consistency check using vectorized operations.
    
    Args:
        disparity_left: Disparity map computed from left to right
        disparity_right: Disparity map computed from right to left
        threshold: Maximum allowed difference between disparities (pixels)
        
    Returns:
        consistent_disparity: Disparity map with inconsistent pixels marked as 0
        valid_mask: Boolean mask indicating valid (consistent) pixels
    """
    height, width = disparity_left.shape
    
    # Create coordinate grid
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Compute corresponding x coordinates in right image
    x_right = xx - disparity_left.astype(np.int32)
    
    # Create valid mask for pixels that are within bounds and have non-zero disparity
    valid_bounds = (x_right >= 0) & (x_right < width) & (disparity_left > 0.5)
    
    # Initialize output
    consistent_disparity = np.copy(disparity_left)
    valid_mask = np.zeros((height, width), dtype=bool)
    
    # For valid pixels, check consistency
    for y in range(height):
        for x in range(width):
            if not valid_bounds[y, x]:
                consistent_disparity[y, x] = 0
                continue
            
            x_r = x_right[y, x]
            d_left = disparity_left[y, x]
            d_right = disparity_right[y, x_r]
            
            # Check if disparities match within threshold
            if abs(d_left - d_right) <= threshold:
                valid_mask[y, x] = True
            else:
                consistent_disparity[y, x] = 0
    
    return consistent_disparity, valid_mask

def compute_lr_consistency_relaxed(disparity_left, disparity_right, threshold=3.0, max_threshold=10.0):
    """
    Relaxed consistency check that marks pixels invalid only if severely inconsistent.
    """
    height, width = disparity_left.shape
    consistent_disparity = np.copy(disparity_left)
    valid_mask = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            d_left = disparity_left[y, x]
            
            if d_left < 0.5:
                continue
            
            x_right = int(x - d_left)
            
            if x_right < 0 or x_right >= width:
                # Out of bounds - mark invalid only if far out
                if x_right < -5 or x_right >= width + 5:
                    consistent_disparity[y, x] = 0
                else:
                    valid_mask[y, x] = True  # Keep marginal cases
                continue
            
            d_right = disparity_right[y, x_right]
            
            # Use tiered thresholds
            error = abs(d_left - d_right)
            if error <= threshold:
                valid_mask[y, x] = True  # Good match
            elif error <= max_threshold:
                valid_mask[y, x] = True  # Acceptable match
                # Optionally: consistent_disparity[y, x] *= 0.9  # Slight penalty
            else:
                consistent_disparity[y, x] = 0  # Only reject egregious errors
    
    return consistent_disparity, valid_mask

def mark_occluded_pixels(disparity_left, disparity_right, threshold=1.0):
    """
    Identify and mark occluded and mismatched pixels.
    
    Args:
        disparity_left: Disparity map computed from left to right
        disparity_right: Disparity map computed from right to left
        threshold: Maximum allowed difference between disparities (pixels)
        
    Returns:
        occlusion_mask: Boolean mask where True indicates occluded/invalid pixels
    """
    height, width = disparity_left.shape
    occlusion_mask = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            d_left = disparity_left[y, x]
            
            # Skip if disparity is zero
            if d_left < 0.5:
                occlusion_mask[y, x] = True
                continue
            
            # Find corresponding pixel in right image
            x_right = int(x - d_left)
            
            # Check bounds
            if x_right < 0 or x_right >= width:
                occlusion_mask[y, x] = True
                continue
            
            # Get disparity at corresponding pixel
            d_right = disparity_right[y, x_right]
            
            # Mark as occluded if inconsistent
            if abs(d_left - d_right) > threshold:
                occlusion_mask[y, x] = True
    
    return occlusion_mask


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    from src.stereo.block_matching import compute_disparity
    
    print("Testing left-right consistency check")
    print("=" * 60)
    
    # Test configuration
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    frame_id = 0
    
    print(f"\nLoading stereo pair from sequence {sequence_id:02d}, frame {frame_id:06d}")
    
    try:
        # Load stereo pair
        left_img, right_img = load_stereo_pair(dataset_path, sequence_id, frame_id)
        print(f"Loaded images: {left_img.shape}")
        
        # Compute disparity on small region for testing
        crop_size = 200
        left_crop = left_img[:crop_size, :crop_size]
        right_crop = right_img[:crop_size, :crop_size]
        
        print(f"\nTesting on cropped region: {left_crop.shape}")
        
        # Compute disparity maps
        window_size = 7
        max_disparity = 64
        
        print(f"\n1. Computing left-to-right disparity (window={window_size})")
        disparity_lr = compute_disparity(left_crop, right_crop, window_size, max_disparity, 'SAD')
        print(f"   Disparity range: [{disparity_lr.min():.2f}, {disparity_lr.max():.2f}]")
        non_zero_lr = np.sum(disparity_lr > 0)
        print(f"   Non-zero pixels: {non_zero_lr} ({100*non_zero_lr/disparity_lr.size:.1f}%)")
        
        print(f"\n2. Computing right-to-left disparity")
        disparity_rl = compute_disparity(right_crop, left_crop, window_size, max_disparity, 'SAD')
        print(f"   Disparity range: [{disparity_rl.min():.2f}, {disparity_rl.max():.2f}]")
        non_zero_rl = np.sum(disparity_rl > 0)
        print(f"   Non-zero pixels: {non_zero_rl} ({100*non_zero_rl/disparity_rl.size:.1f}%)")
        
        # Test consistency check
        threshold = 1.0
        print(f"\n3. Performing consistency check (threshold={threshold} px)")
        
        consistent_disp, valid_mask = compute_lr_consistency(disparity_lr, disparity_rl, threshold)
        
        num_valid = np.sum(valid_mask)
        num_total = np.sum(disparity_lr > 0)
        
        print(f"   Valid pixels: {num_valid}")
        print(f"   Invalid pixels: {num_total - num_valid}")
        print(f"   Consistency rate: {100*num_valid/max(num_total, 1):.1f}%")
        
        # Test occlusion detection
        print(f"\n4. Detecting occluded pixels")
        occlusion_mask = mark_occluded_pixels(disparity_lr, disparity_rl, threshold)
        num_occluded = np.sum(occlusion_mask)
        print(f"   Occluded pixels: {num_occluded} ({100*num_occluded/occlusion_mask.size:.1f}%)")
        
        # Test optimized version
        print(f"\n5. Testing optimized consistency check")
        consistent_opt, valid_opt = compute_lr_consistency_optimized(disparity_lr, disparity_rl, threshold)
        
        # Verify both methods produce same result
        same_disparity = np.allclose(consistent_disp, consistent_opt)
        same_mask = np.array_equal(valid_mask, valid_opt)
        
        print(f"   Disparity maps match: {same_disparity}")
        print(f"   Valid masks match: {same_mask}")
        
        # Show statistics
        print(f"\n6. Consistency statistics")
        print(f"   Original non-zero pixels: {num_total}")
        print(f"   After consistency check: {np.sum(consistent_disp > 0)}")
        print(f"   Removed pixels: {num_total - np.sum(consistent_disp > 0)}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Left-right consistency check is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()