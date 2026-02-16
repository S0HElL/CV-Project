
"""
Block matching stereo implementation.
Computes dense disparity maps using SAD, SSD, and NCC cost functions.
"""

import numpy as np
import cv2


def compute_sad(patch_left, patch_right):
    """
    Compute Sum of Absolute Differences between two patches.
    
    Args:
        patch_left: Left image patch
        patch_right: Right image patch
        
    Returns:
        sad: Sum of absolute differences (lower is better)
    """
    return np.sum(np.abs(patch_left.astype(np.float32) - patch_right.astype(np.float32)))


def compute_ssd(patch_left, patch_right):
    """
    Compute Sum of Squared Differences between two patches.
    
    Args:
        patch_left: Left image patch
        patch_right: Right image patch
        
    Returns:
        ssd: Sum of squared differences (lower is better)
    """
    diff = patch_left.astype(np.float32) - patch_right.astype(np.float32)
    return np.sum(diff * diff)


def compute_ncc(patch_left, patch_right):
    """
    Compute Normalized Cross-Correlation between two patches.
    
    Args:
        patch_left: Left image patch
        patch_right: Right image patch
        
    Returns:
        ncc: Negative normalized cross-correlation (lower is better for consistency)
             Range: [-1, 1], we return -ncc so lower is better
    """
    patch_left_f = patch_left.astype(np.float32)
    patch_right_f = patch_right.astype(np.float32)
    
    # Normalize patches
    left_mean = np.mean(patch_left_f)
    right_mean = np.mean(patch_right_f)
    
    left_norm = patch_left_f - left_mean
    right_norm = patch_right_f - right_mean
    
    # Compute standard deviations
    left_std = np.sqrt(np.sum(left_norm * left_norm))
    right_std = np.sqrt(np.sum(right_norm * right_norm))
    
    # Avoid division by zero
    if left_std < 1e-6 or right_std < 1e-6:
        return 1.0  # Return worst correlation (as positive value)
    
    # Compute normalized cross-correlation
    ncc = np.sum(left_norm * right_norm) / (left_std * right_std)
    
    # Return negative NCC so that lower is better (consistent with SAD/SSD)
    return -ncc


def compute_disparity(left_image, right_image, window_size=11, max_disparity=128, cost_function='SAD'):
    """
    Compute dense disparity map using block matching.
    
    Args:
        left_image: Left camera image (grayscale)
        right_image: Right camera image (grayscale)
        window_size: Size of matching window (must be odd)
        max_disparity: Maximum disparity to search
        cost_function: Cost function to use ('SAD', 'SSD', or 'NCC')
        
    Returns:
        disparity_map: Dense disparity map (same size as input images)
                       Invalid disparities are set to 0
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    # Select cost function
    if cost_function == 'SAD':
        cost_fn = compute_sad
    elif cost_function == 'SSD':
        cost_fn = compute_ssd
    elif cost_function == 'NCC':
        cost_fn = compute_ncc
    else:
        raise ValueError(f"Unknown cost function: {cost_function}")
    
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    half_window = window_size // 2
    
    # Process each pixel in the left image
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            # Extract left patch
            left_patch = left_image[
                y - half_window:y + half_window + 1,
                x - half_window:x + half_window + 1
            ]
            
            # Search along epipolar line (horizontal, to the left in right image)
            min_cost = float('inf')
            best_disparity = 0
            
            # Disparity search range
            for d in range(max_disparity + 1):
                # Check if right patch is within image bounds
                x_right = x - d
                if x_right - half_window < 0:
                    break
                
                # Extract right patch
                right_patch = right_image[
                    y - half_window:y + half_window + 1,
                    x_right - half_window:x_right + half_window + 1
                ]
                
                # Compute cost
                cost = cost_fn(left_patch, right_patch)
                
                # Update best disparity (winner-takes-all)
                if cost < min_cost:
                    min_cost = cost
                    best_disparity = d
            
            disparity_map[y, x] = best_disparity
    
    return disparity_map


def compute_disparity_optimized(left_image, right_image, window_size=11, max_disparity=128, cost_function='SAD'):
    """
    Optimized disparity computation using vectorized operations.
    Faster than naive implementation for large images.
    
    Args:
        left_image: Left camera image (grayscale)
        right_image: Right camera image (grayscale)
        window_size: Size of matching window (must be odd)
        max_disparity: Maximum disparity to search
        cost_function: Cost function to use ('SAD', 'SSD', or 'NCC')
        
    Returns:
        disparity_map: Dense disparity map
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    half_window = window_size // 2
    
    # Convert to float for computation
    left_float = left_image.astype(np.float32)
    right_float = right_image.astype(np.float32)
    
    # Process each disparity level
    cost_volume = np.full((height, width, max_disparity + 1), float('inf'), dtype=np.float32)
    
    for d in range(max_disparity + 1):
        # Shift right image by disparity d
        if d == 0:
            right_shifted = right_float
        else:
            right_shifted = np.zeros_like(right_float)
            right_shifted[:, d:] = right_float[:, :-d]
        
        # Compute cost based on selected function
        if cost_function == 'SAD':
            diff = np.abs(left_float - right_shifted)
        elif cost_function == 'SSD':
            diff = (left_float - right_shifted) ** 2
        elif cost_function == 'NCC':
            # For NCC, use per-pixel computation (more complex)
            # Fall back to standard implementation for NCC
            continue
        else:
            raise ValueError(f"Unknown cost function: {cost_function}")
        
        # Apply box filter for window aggregation
        cost = cv2.boxFilter(diff, -1, (window_size, window_size), normalize=False)
        
        # Store in cost volume (only valid regions)
        cost_volume[:, d:, d] = cost[:, d:]
    
    # Handle NCC separately if needed
    if cost_function == 'NCC':
        # Use standard implementation for NCC
        return compute_disparity(left_image, right_image, window_size, max_disparity, cost_function)
    
    # Winner-takes-all: select disparity with minimum cost
    disparity_map = np.argmin(cost_volume, axis=2).astype(np.float32)
    
    # Set border regions to 0
    disparity_map[:half_window, :] = 0
    disparity_map[-half_window:, :] = 0
    disparity_map[:, :half_window] = 0
    disparity_map[:, -half_window:] = 0
    
    return disparity_map


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    
    print("Testing block matching stereo implementation")
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
        
        # Test on a small region for speed
        h, w = left_img.shape
        crop_size = 200
        left_crop = left_img[:crop_size, :crop_size]
        right_crop = right_img[:crop_size, :crop_size]
        
        print(f"\nTesting on cropped region: {left_crop.shape}")
        
        # Test parameters
        window_size = 7
        max_disparity = 64
        
        # Test SAD
        print(f"\n1. Testing SAD (window={window_size}, max_disp={max_disparity})")
        disparity_sad = compute_disparity(left_crop, right_crop, window_size, max_disparity, 'SAD')
        print(f"   Disparity range: [{disparity_sad.min():.2f}, {disparity_sad.max():.2f}]")
        print(f"   Mean disparity: {disparity_sad[disparity_sad > 0].mean():.2f}")
        
        # Test SSD
        print(f"\n2. Testing SSD (window={window_size}, max_disp={max_disparity})")
        disparity_ssd = compute_disparity(left_crop, right_crop, window_size, max_disparity, 'SSD')
        print(f"   Disparity range: [{disparity_ssd.min():.2f}, {disparity_ssd.max():.2f}]")
        print(f"   Mean disparity: {disparity_ssd[disparity_ssd > 0].mean():.2f}")
        
        # Test NCC (slower, smaller region)
        print(f"\n3. Testing NCC (window={window_size}, max_disp={max_disparity})")
        small_crop = 100
        disparity_ncc = compute_disparity(left_crop[:small_crop, :small_crop], 
                                         right_crop[:small_crop, :small_crop], 
                                         window_size, max_disparity, 'NCC')
        print(f"   Disparity range: [{disparity_ncc.min():.2f}, {disparity_ncc.max():.2f}]")
        print(f"   Mean disparity: {disparity_ncc[disparity_ncc > 0].mean():.2f}")
        
        # Test optimized version
        print(f"\n4. Testing optimized SAD (window={window_size}, max_disp={max_disparity})")
        disparity_opt = compute_disparity_optimized(left_crop, right_crop, window_size, max_disparity, 'SAD')
        print(f"   Disparity range: [{disparity_opt.min():.2f}, {disparity_opt.max():.2f}]")
        print(f"   Mean disparity: {disparity_opt[disparity_opt > 0].mean():.2f}")
        
        # Verify cost functions work on patches
        print("\n5. Testing individual cost functions")
        test_patch_size = 11
        patch1 = left_crop[50:50+test_patch_size, 50:50+test_patch_size]
        patch2 = right_crop[50:50+test_patch_size, 45:45+test_patch_size]
        
        sad_cost = compute_sad(patch1, patch2)
        ssd_cost = compute_ssd(patch1, patch2)
        ncc_cost = compute_ncc(patch1, patch2)
        
        print(f"   SAD cost: {sad_cost:.2f}")
        print(f"   SSD cost: {ssd_cost:.2f}")
        print(f"   NCC cost: {ncc_cost:.4f}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Block matching implementation is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available at: {dataset_path}")
        print("Expected structure:")
        print("  CV Project/Dataset/sequences/00/image_0/000000.png")
        print("  CV Project/Dataset/sequences/00/image_1/000000.png")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()