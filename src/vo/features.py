"""
Feature detection and matching for visual odometry.
"""

import numpy as np
import cv2


def detect_features(image, detector_type='ORB', max_features=3000):
    """
    Detect keypoints and compute descriptors in an image.
    
    Args:
        image: Input grayscale image
        detector_type: Type of feature detector ('ORB' or 'SIFT')
        max_features: Maximum number of features to detect
        
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: NumPy array of feature descriptors
    """
    if detector_type == 'ORB':
        # Create ORB detector
        detector = cv2.ORB_create(nfeatures=max_features)
    elif detector_type == 'SIFT':
        # Create SIFT detector
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(image, None)
    
    return keypoints, descriptors


def match_features(descriptors1, descriptors2, detector_type='ORB', ratio_threshold=0.75):
    """
    Match feature descriptors between two images.
    
    Args:
        descriptors1: Descriptors from first image
        descriptors2: Descriptors from second image
        detector_type: Type of detector used ('ORB' or 'SIFT')
        ratio_threshold: Lowe's ratio test threshold (for SIFT)
        
    Returns:
        matches: List of cv2.DMatch objects representing good matches
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return []
    
    if detector_type == 'ORB':
        # Use Hamming distance for binary descriptors (ORB)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Use KNN matching with k=2 for ratio test
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    matches.append(m)
            elif len(match_pair) == 1:
                # Only one match found, accept it
                matches.append(match_pair[0])
        
    elif detector_type == 'SIFT':
        # Use L2 distance for SIFT descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Use KNN matching with k=2 for ratio test
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    matches.append(m)
            elif len(match_pair) == 1:
                matches.append(match_pair[0])
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return matches


def extract_matched_points(keypoints1, keypoints2, matches):
    """
    Extract corresponding point coordinates from matches.
    
    Args:
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of cv2.DMatch objects
        
    Returns:
        points1: Nx2 array of (x, y) coordinates in first image
        points2: Nx2 array of (x, y) coordinates in second image
    """
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt
    
    return points1, points2


def detect_and_match(image1, image2, detector_type='ORB', max_features=3000, ratio_threshold=0.75):
    """
    Complete pipeline: detect features and match between two images.
    
    Args:
        image1: First grayscale image
        image2: Second grayscale image
        detector_type: Type of feature detector ('ORB' or 'SIFT')
        max_features: Maximum number of features to detect
        ratio_threshold: Lowe's ratio test threshold
        
    Returns:
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of good matches
        points1: Matched point coordinates in first image
        points2: Matched point coordinates in second image
    """
    # Detect features in both images
    keypoints1, descriptors1 = detect_features(image1, detector_type, max_features)
    keypoints2, descriptors2 = detect_features(image2, detector_type, max_features)
    
    # Match features
    matches = match_features(descriptors1, descriptors2, detector_type, ratio_threshold)
    
    # Extract matched point coordinates
    if len(matches) > 0:
        points1, points2 = extract_matched_points(keypoints1, keypoints2, matches)
    else:
        points1 = np.array([])
        points2 = np.array([])
    
    return keypoints1, keypoints2, matches, points1, points2


def visualize_matches(image1, image2, keypoints1, keypoints2, matches, max_display=100):
    """
    Visualize feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of matches
        max_display: Maximum number of matches to display
        
    Returns:
        vis_image: Visualization image showing matches
    """
    # Limit number of matches for visualization
    display_matches = matches[:max_display]
    
    # Draw matches
    vis_image = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        display_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
    
    return vis_image


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.utils.kitti_odometry_loader import load_left_image
    
    print("Testing feature detection and matching")
    print("=" * 60)
    
    # Test configuration
    dataset_path = "..\\CV Project\\Dataset"
    sequence_id = 0
    frame1_id = 0
    frame2_id = 1
    
    print(f"\nLoading consecutive frames from sequence {sequence_id:02d}")
    
    try:
        # Load consecutive frames
        print(f"\n1. Loading images")
        image1 = load_left_image(dataset_path, sequence_id, frame1_id)
        image2 = load_left_image(dataset_path, sequence_id, frame2_id)
        print(f"   Image 1 shape: {image1.shape}")
        print(f"   Image 2 shape: {image2.shape}")
        
        # Test ORB detector
        print(f"\n2. Testing ORB feature detection")
        kp1_orb, desc1_orb = detect_features(image1, detector_type='ORB', max_features=2000)
        kp2_orb, desc2_orb = detect_features(image2, detector_type='ORB', max_features=2000)
        print(f"   Frame 1: {len(kp1_orb)} keypoints")
        print(f"   Frame 2: {len(kp2_orb)} keypoints")
        print(f"   Descriptor shape: {desc1_orb.shape}")
        
        # Test ORB matching
        print(f"\n3. Testing ORB feature matching")
        matches_orb = match_features(desc1_orb, desc2_orb, detector_type='ORB', ratio_threshold=0.75)
        print(f"   Total matches: {len(matches_orb)}")
        
        if len(matches_orb) > 0:
            # Show match distance statistics
            distances = [m.distance for m in matches_orb]
            print(f"   Distance range: [{min(distances):.1f}, {max(distances):.1f}]")
            print(f"   Mean distance: {np.mean(distances):.1f}")
        
        # Test point extraction
        print(f"\n4. Testing point extraction")
        if len(matches_orb) > 0:
            pts1_orb, pts2_orb = extract_matched_points(kp1_orb, kp2_orb, matches_orb)
            print(f"   Points 1 shape: {pts1_orb.shape}")
            print(f"   Points 2 shape: {pts2_orb.shape}")
            
            # Show some example points
            print(f"   Example matches:")
            for i in range(min(3, len(matches_orb))):
                print(f"     Match {i}: ({pts1_orb[i, 0]:.1f}, {pts1_orb[i, 1]:.1f}) -> "
                      f"({pts2_orb[i, 0]:.1f}, {pts2_orb[i, 1]:.1f})")
        
        # Test SIFT detector (if available)
        print(f"\n5. Testing SIFT feature detection")
        try:
            kp1_sift, desc1_sift = detect_features(image1, detector_type='SIFT', max_features=2000)
            kp2_sift, desc2_sift = detect_features(image2, detector_type='SIFT', max_features=2000)
            print(f"   Frame 1: {len(kp1_sift)} keypoints")
            print(f"   Frame 2: {len(kp2_sift)} keypoints")
            print(f"   Descriptor shape: {desc1_sift.shape}")
            
            # Test SIFT matching
            print(f"\n6. Testing SIFT feature matching")
            matches_sift = match_features(desc1_sift, desc2_sift, detector_type='SIFT', ratio_threshold=0.75)
            print(f"   Total matches: {len(matches_sift)}")
            
        except cv2.error as e:
            print(f"   SIFT not available (may require opencv-contrib-python): {e}")
        
        # Test complete pipeline
        print(f"\n7. Testing complete detect_and_match pipeline")
        kp1, kp2, matches, pts1, pts2 = detect_and_match(
            image1, image2, 
            detector_type='ORB',
            max_features=2000,
            ratio_threshold=0.75
        )
        print(f"   Detected {len(kp1)} and {len(kp2)} keypoints")
        print(f"   Found {len(matches)} matches")
        print(f"   Extracted {len(pts1)} point pairs")
        
        # Test visualization
        print(f"\n8. Testing match visualization")
        vis_img = visualize_matches(image1, image2, kp1, kp2, matches, max_display=50)
        print(f"   Visualization image shape: {vis_img.shape}")
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("Feature detection and matching is working correctly.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure KITTI dataset is available")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
