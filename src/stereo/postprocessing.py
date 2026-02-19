import numpy as np
import cv2

def postprocess_disparity(disparity_map, valid_mask=None, median_kernel=5, 
                         fill_method='horizontal', use_bilateral=False,
                         remove_speckles=True, min_region_size=50):
    """
    Optimized pipeline with original function signature.
    """
    processed = np.copy(disparity_map)
    
    # 1. Remove Speckles (Noise)
    if remove_speckles:
        # Binarize
        mask = (processed > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_region_size:
                processed[labels == i] = 0
    
    # 2. Fill Holes (Vectorized Logic)
    # We use your 'fill_method' argument to decide the strategy
    if fill_method == 'horizontal':
        processed = _vectorized_horizontal_fill(processed)
    elif fill_method == 'nearest' or fill_method == 'inpaint':
        # Fallback to OpenCV inpaint for complex methods (much faster than loops)
        mask = (processed <= 0).astype(np.uint8)
        # Scale to uint8 for cv2.inpaint
        d_max = processed.max() if processed.max() > 0 else 1
        disp_u8 = (processed * 255.0 / d_max).astype(np.uint8)
        inpainted = cv2.inpaint(disp_u8, mask, 3, cv2.INPAINT_TELEA)
        processed = inpainted.astype(np.float32) * d_max / 255.0

    # 3. Median Filter
    if median_kernel > 1:
        # Use uint16 to preserve sub-pixel precision better than uint8
        scale = 256.0
        disp_u16 = (processed * scale).astype(np.uint16)
        filtered_u16 = cv2.medianBlur(disp_u16, median_kernel)
        processed = filtered_u16.astype(np.float32) / scale

    # 4. Optional Bilateral
    if use_bilateral:
        d_max = processed.max() if processed.max() > 0 else 1
        disp_u8 = (processed * 255.0 / d_max).astype(np.uint8)
        processed = cv2.bilateralFilter(disp_u8, 9, 75, 75).astype(np.float32) * d_max / 255.0

    return processed

def _vectorized_horizontal_fill(disparity):
    """Helper for lightning-fast horizontal propagation."""
    h, w = disparity.shape
    idx_grid = np.tile(np.arange(w), (h, 1))
    
    # Propagate Left to Right
    l_mask = disparity > 0
    l_idx = np.where(l_mask, idx_grid, 0)
    l_filled_idx = np.maximum.accumulate(l_idx, axis=1)
    fill_l = disparity[np.arange(h)[:, None], l_filled_idx]
    
    # Propagate Right to Left
    r_mask = disparity[:, ::-1] > 0
    r_idx = np.where(r_mask, idx_grid, 0)
    r_filled_idx = np.maximum.accumulate(r_idx, axis=1)
    fill_r = disparity[:, ::-1][np.arange(h)[:, None], r_filled_idx][:, ::-1]
    
    # Combine (Take average of both directions for smoother fill)
    # or take the one that isn't zero
    res = np.where((fill_l > 0) & (fill_r > 0), (fill_l + fill_r) / 2, np.maximum(fill_l, fill_r))
    return res