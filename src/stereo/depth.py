#depth = (f * B) / disparity
import numpy as np
def disparity_to_depth(disparity_map, focal_length, baseline, min_disparity=1.0, max_depth=80.0):
    """
    Convert disparity map to depth map.
    
    Args:
        disparity_map: Disparity map (pixels)
        focal_length: Camera focal length f (pixels)
        baseline: Stereo baseline B (meters)
        min_disparity: Minimum valid disparity (default 1.0 pixels)
        max_depth: Maximum valid depth (default 80 meters)
    
    Returns:
        depth_map: Depth map (meters)
    """ 
    # Create output depth map
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    
    # Only compute depth where disparity is valid and above threshold
    valid_mask = disparity_map >= min_disparity
    
    # Compute depth
    depth_map[valid_mask] = (focal_length * baseline) / disparity_map[valid_mask]
    
    # Filter out unrealistic depths
    depth_map[depth_map > max_depth] = 0
    
    return depth_map

if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_stereo_loader import load_stereo_pair
    from src.utils.kitti_odometry_loader import load_calibration
    from src.stereo.block_matching import compute_disparity
    
    print("Testing depth computation from disparity")
    print("=" * 60)
    
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    frame_id = 0
    
    try:
        # Step 1: Load stereo pair
        print(f"\n1. Loading stereo pair from sequence {sequence_id:02d}, frame {frame_id:06d}")
        left_img, right_img = load_stereo_pair(dataset_path, sequence_id, frame_id)
        print(f"   Image shape: {left_img.shape}")
        
        # Step 2: Load calibration
        print(f"\n2. Loading calibration data")
        calib = load_calibration(dataset_path, sequence_id)
        focal_length = calib['f']
        baseline = calib['B']
        print(f"   Focal length: {focal_length:.2f} pixels")
        print(f"   Baseline: {baseline:.4f} meters")
        
        # Step 3: Compute disparity (on small crop for speed)
        print(f"\n3. Computing disparity map")
        crop_size = 200
        left_crop = left_img[:crop_size, :crop_size]
        right_crop = right_img[:crop_size, :crop_size]
        
        disparity = compute_disparity(left_crop, right_crop, window_size=7, max_disparity=64, cost_function='SAD')
        valid_disp_pixels = np.sum(disparity > 0)
        print(f"   Disparity range: [{disparity.min():.1f}, {disparity.max():.1f}]")
        print(f"   Valid pixels: {valid_disp_pixels}")
        
        # Step 4: Convert to depth
        print(f"\n4. Converting disparity to depth")
        depth_map = disparity_to_depth(disparity, focal_length, baseline)
        
        # Step 5: Show depth statistics
        print(f"\n5. Depth map statistics")
        valid_depth = depth_map[depth_map > 0]
        
        if len(valid_depth) > 0:
            print(f"   Valid depth pixels: {len(valid_depth)}")
            print(f"   Depth range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] meters")
            print(f"   Mean depth: {valid_depth.mean():.2f} meters")
            print(f"   Median depth: {np.median(valid_depth):.2f} meters")
        else:
            print(f"   No valid depth pixels!")
        
        # Step 6: Verify depth formula
        print(f"\n6. Verifying depth computation")
        # Pick a pixel with valid disparity
        test_y, test_x = 100, 100
        if disparity[test_y, test_x] > 0:
            test_disp = disparity[test_y, test_x]
            test_depth = depth_map[test_y, test_x]
            expected_depth = focal_length * baseline / test_disp
            
            print(f"   Test pixel ({test_y}, {test_x}):")
            print(f"     Disparity: {test_disp:.2f} pixels")
            print(f"     Depth: {test_depth:.2f} meters")
            print(f"     Expected: {expected_depth:.2f} meters")
            print(f"     Match: {np.isclose(test_depth, expected_depth)}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Depth computation is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()