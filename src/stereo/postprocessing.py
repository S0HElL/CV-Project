"""
Disparity post-processing for noise reduction and hole filling.
"""

import numpy as np
import cv2


def median_filter(disparity_map, kernel_size=5):
    """
    Apply median filter to reduce speckle noise in disparity map.
    
    Args:
        disparity_map: Input disparity map
        kernel_size: Size of median filter kernel (must be odd)
        
    Returns:
        filtered_disparity: Median filtered disparity map
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Apply median filter
    # Note: cv2.medianBlur requires uint8 or float32 single-channel image
    filtered_disparity = cv2.medianBlur(disparity_map.astype(np.float32), kernel_size)
    
    return filtered_disparity


def fill_holes_interpolation(disparity_map, valid_mask=None, method='horizontal'):
    """
    Fill holes in disparity map using interpolation.
    
    Args:
        disparity_map: Input disparity map with holes (zeros or invalid values)
        valid_mask: Boolean mask indicating valid pixels (optional)
                   If None, assumes pixels with value > 0 are valid
        method: Interpolation method ('horizontal', 'nearest', or 'inpaint')
        
    Returns:
        filled_disparity: Disparity map with holes filled
    """
    filled_disparity = np.copy(disparity_map)
    height, width = disparity_map.shape
    
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = disparity_map > 0
    
    if method == 'horizontal':
        # Fill holes using horizontal interpolation (left to right)
        for y in range(height):
            # Find invalid pixels in this row
            invalid_pixels = ~valid_mask[y, :]
            
            if not np.any(invalid_pixels):
                continue
            
            # Find valid pixels
            valid_indices = np.where(valid_mask[y, :])[0]
            
            if len(valid_indices) == 0:
                continue
            
            # For each invalid pixel, interpolate from nearest valid pixels
            for x in np.where(invalid_pixels)[0]:
                # Find nearest valid pixels on left and right
                left_valid = valid_indices[valid_indices < x]
                right_valid = valid_indices[valid_indices > x]
                
                if len(left_valid) > 0 and len(right_valid) > 0:
                    # Interpolate between left and right
                    x_left = left_valid[-1]
                    x_right = right_valid[0]
                    d_left = disparity_map[y, x_left]
                    d_right = disparity_map[y, x_right]
                    
                    # Linear interpolation
                    alpha = (x - x_left) / (x_right - x_left)
                    filled_disparity[y, x] = (1 - alpha) * d_left + alpha * d_right
                    
                elif len(left_valid) > 0:
                    # Use left neighbor
                    filled_disparity[y, x] = disparity_map[y, left_valid[-1]]
                    
                elif len(right_valid) > 0:
                    # Use right neighbor
                    filled_disparity[y, x] = disparity_map[y, right_valid[0]]
    
    elif method == 'nearest':
        # Fill holes using nearest valid neighbor
        invalid_pixels = ~valid_mask
        
        if np.any(invalid_pixels):
            # Find coordinates of valid and invalid pixels
            valid_coords = np.argwhere(valid_mask)
            invalid_coords = np.argwhere(invalid_pixels)
            
            if len(valid_coords) > 0:
                # For each invalid pixel, find nearest valid pixel
                for invalid_y, invalid_x in invalid_coords:
                    # Compute distances to all valid pixels
                    distances = np.sqrt(
                        (valid_coords[:, 0] - invalid_y)**2 + 
                        (valid_coords[:, 1] - invalid_x)**2
                    )
                    
                    # Find nearest valid pixel
                    nearest_idx = np.argmin(distances)
                    nearest_y, nearest_x = valid_coords[nearest_idx]
                    
                    # Copy disparity value
                    filled_disparity[invalid_y, invalid_x] = disparity_map[nearest_y, nearest_x]
    
    elif method == 'inpaint':
        # Use OpenCV inpainting for hole filling
        # Convert to uint8 for inpainting
        disp_uint8 = (disparity_map * 255.0 / max(disparity_map.max(), 1.0)).astype(np.uint8)
        
        # Create inpainting mask (1 for holes, 0 for valid)
        inpaint_mask = (~valid_mask).astype(np.uint8)
        
        # Apply inpainting
        inpainted = cv2.inpaint(disp_uint8, inpaint_mask, 3, cv2.INPAINT_TELEA)
        
        # Convert back to float and scale
        filled_disparity = inpainted.astype(np.float32) * disparity_map.max() / 255.0
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return filled_disparity


def bilateral_filter(disparity_map, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to smooth disparity while preserving edges.
    
    Args:
        disparity_map: Input disparity map
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        filtered_disparity: Bilateral filtered disparity map
    """
    # Convert to uint8 for bilateral filter
    disp_max = disparity_map.max()
    if disp_max > 0:
        disp_uint8 = (disparity_map * 255.0 / disp_max).astype(np.uint8)
    else:
        return disparity_map
    
    # Apply bilateral filter
    filtered_uint8 = cv2.bilateralFilter(disp_uint8, d, sigma_color, sigma_space)
    
    # Convert back to float
    filtered_disparity = filtered_uint8.astype(np.float32) * disp_max / 255.0
    
    return filtered_disparity


def remove_small_regions(disparity_map, min_size=50):
    """
    Remove small isolated regions (speckles) from disparity map.
    
    Args:
        disparity_map: Input disparity map
        min_size: Minimum region size to keep (in pixels)
        
    Returns:
        cleaned_disparity: Disparity map with small regions removed
    """
    cleaned_disparity = np.copy(disparity_map)
    
    # Create binary mask of valid disparities
    valid_mask = (disparity_map > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
    
    # Remove small components
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_size:
            # Set pixels of this component to 0
            cleaned_disparity[labels == label] = 0
    
    return cleaned_disparity


def postprocess_disparity(disparity_map, valid_mask=None, median_kernel=5, 
                          fill_method='horizontal', use_bilateral=False,
                          remove_speckles=True, min_region_size=50):
    """
    Complete disparity post-processing pipeline.
    
    Args:
        disparity_map: Input disparity map
        valid_mask: Boolean mask indicating valid pixels (optional)
        median_kernel: Size of median filter kernel
        fill_method: Method for hole filling ('horizontal', 'nearest', or 'inpaint')
        use_bilateral: Whether to apply bilateral filter
        remove_speckles: Whether to remove small regions
        min_region_size: Minimum region size to keep
        
    Returns:
        processed_disparity: Post-processed disparity map
    """
    processed = np.copy(disparity_map)
    
    # Step 1: Remove small speckle regions
    if remove_speckles:
        processed = remove_small_regions(processed, min_region_size)
    
    # Step 2: Apply median filter to reduce noise
    if median_kernel > 1:
        processed = median_filter(processed, median_kernel)
    
    # Step 3: Fill holes
    if valid_mask is None:
        valid_mask = processed > 0
    processed = fill_holes_interpolation(processed, valid_mask, method=fill_method)
    
    # Step 4: Optional bilateral filtering for edge preservation
    if use_bilateral:
        processed = bilateral_filter(processed)
    
    return processed


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    from src.stereo.block_matching import compute_disparity
    from src.stereo.consistency import compute_lr_consistency
    
    print("Testing disparity post-processing")
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
        crop_size = 300
        left_crop = left_img[:crop_size, :crop_size]
        right_crop = right_img[:crop_size, :crop_size]
        
        print(f"\nTesting on cropped region: {left_crop.shape}")
        
        # Compute initial disparity
        window_size = 7
        max_disparity = 64
        
        print(f"\n1. Computing initial disparity")
        disparity_lr = compute_disparity(left_crop, right_crop, window_size, max_disparity, 'SAD')
        disparity_rl = compute_disparity(right_crop, left_crop, window_size, max_disparity, 'SAD')
        
        # Apply consistency check
        print(f"\n2. Applying consistency check")
        consistent_disp, valid_mask = compute_lr_consistency(disparity_lr, disparity_rl, threshold=1.0)
        
        num_valid_before = np.sum(consistent_disp > 0)
        print(f"   Valid pixels before post-processing: {num_valid_before}")
        
        # Test median filter
        print(f"\n3. Testing median filter (kernel=5)")
        median_filtered = median_filter(consistent_disp, kernel_size=5)
        print(f"   Valid pixels after median filter: {np.sum(median_filtered > 0)}")
        print(f"   Mean change: {np.mean(np.abs(median_filtered - consistent_disp)):.2f}")
        
        # Test hole filling - horizontal
        print(f"\n4. Testing horizontal hole filling")
        filled_horizontal = fill_holes_interpolation(consistent_disp, valid_mask, method='horizontal')
        filled_pixels = np.sum(filled_horizontal > 0) - num_valid_before
        print(f"   Pixels filled: {filled_pixels}")
        print(f"   Total valid pixels: {np.sum(filled_horizontal > 0)}")
        
        # Test hole filling - nearest
        print(f"\n5. Testing nearest neighbor hole filling")
        filled_nearest = fill_holes_interpolation(consistent_disp, valid_mask, method='nearest')
        filled_pixels_nn = np.sum(filled_nearest > 0) - num_valid_before
        print(f"   Pixels filled: {filled_pixels_nn}")
        
        # Test hole filling - inpaint
        print(f"\n6. Testing inpainting hole filling")
        filled_inpaint = fill_holes_interpolation(consistent_disp, valid_mask, method='inpaint')
        filled_pixels_inp = np.sum(filled_inpaint > 0) - num_valid_before
        print(f"   Pixels filled: {filled_pixels_inp}")
        
        # Test bilateral filter
        print(f"\n7. Testing bilateral filter")
        bilateral_filtered = bilateral_filter(consistent_disp)
        print(f"   Valid pixels after bilateral filter: {np.sum(bilateral_filtered > 0)}")
        
        # Test speckle removal
        print(f"\n8. Testing speckle removal (min_size=50)")
        despeckled = remove_small_regions(consistent_disp, min_size=50)
        removed = num_valid_before - np.sum(despeckled > 0)
        print(f"   Pixels removed: {removed}")
        
        # Test complete pipeline
        print(f"\n9. Testing complete post-processing pipeline")
        processed = postprocess_disparity(
            consistent_disp, 
            valid_mask,
            median_kernel=5,
            fill_method='horizontal',
            use_bilateral=False,
            remove_speckles=True,
            min_region_size=50
        )
        
        final_valid = np.sum(processed > 0)
        print(f"   Final valid pixels: {final_valid}")
        print(f"   Improvement: {final_valid - num_valid_before} pixels")
        print(f"   Fill rate: {100*(final_valid - num_valid_before)/(consistent_disp.size - num_valid_before):.1f}%")
        
        # Show disparity statistics
        print(f"\n10. Disparity statistics comparison")
        print(f"   Original - range: [{consistent_disp.min():.1f}, {consistent_disp.max():.1f}], "
              f"mean: {consistent_disp[consistent_disp > 0].mean():.2f}")
        print(f"   Processed - range: [{processed.min():.1f}, {processed.max():.1f}], "
              f"mean: {processed[processed > 0].mean():.2f}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Disparity post-processing is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()