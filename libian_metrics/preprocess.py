"""
Preprocessing pipeline for ancient character glyphs.

Main function:
    preprocess: Complete preprocessing pipeline including denoising, binarization,
                deskewing, rescaling, and skeletonization.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

from .io_utils import (
    remove_small_components,
    estimate_skew_angle,
    rotate_bound,
    crop_to_content,
    pad_to_size
)
from .skeleton import skeletonize, prune_spurs


def preprocess(
    img_bgr: np.ndarray,
    target_height: int = 256,
    small_comp_ratio: float = 2e-4
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for ancient character glyphs.
    
    Args:
        img_bgr: Input image in BGR format
        target_height: Target height for rescaling (maintains aspect ratio)
        small_comp_ratio: Ratio of image area to remove small components
        
    Returns:
        (bin_img, skel, meta) where:
        - bin_img: Binary image (foreground=255, background=0)
        - skel: Skeleton image (0/255)
        - meta: Dictionary with keys:
            - 'angle': Estimated rotation angle in degrees
            - 'bbox': Bounding box (y0, y1, x0, x1)
            - 'scale': Scale factor applied
            - 'quality_flag': Whether image passes quality checks
    """
    
    meta = {}
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Adaptive binarization
    bin_img = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10
    )
    
    # Step 3: Remove small components
    img_area = bin_img.shape[0] * bin_img.shape[1]
    min_area = max(10, int(img_area * small_comp_ratio))
    bin_img = remove_small_components(bin_img, min_area)
    
    # Step 4: Estimate and correct skew
    angle = estimate_skew_angle(bin_img, angle_range=5.0)
    meta['angle'] = float(angle)
    
    if abs(angle) > 0.5:
        # Apply rotation
        bin_img = rotate_bound(bin_img, angle)
        # Convert back to binary if needed
        if len(bin_img.shape) == 3:
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Step 5: Crop to content
    bin_img, bbox = crop_to_content(bin_img)
    meta['bbox'] = [int(b) for b in bbox]
    
    # Step 6: Rescale to target height
    h, w = bin_img.shape[:2]
    if h > 0:
        scale = target_height / h
        new_w = max(1, int(w * scale))
        bin_img = cv2.resize(bin_img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
    
    meta['scale'] = float(scale)
    
    # Step 7: Pad to square (for consistency)
    max_dim = max(bin_img.shape[:2])
    padded, _ = pad_to_size(bin_img, max_dim)
    bin_img = padded
    
    # Step 8: Skeletonize
    skel = skeletonize(bin_img)
    
    # Step 9: Prune spurs
    skel = prune_spurs(skel, min_branch_len=6)
    
    # Step 10: Quality checks
    max_cc = int(_get_largest_component_area(bin_img))
    skel_pixels = int(cv2.countNonZero(skel))
    
    quality_flag = True
    if max_cc < img_area * 0.01:  # Max component < 1% of image area
        quality_flag = False
    if skel_pixels < 200:  # Too few skeleton pixels
        quality_flag = False
    
    meta['quality_flag'] = bool(quality_flag)
    meta['skel_pixels'] = skel_pixels
    meta['max_component_area'] = max_cc
    
    return bin_img, skel, meta


def _get_largest_component_area(bin_img: np.ndarray) -> int:
    """Get area of largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    
    if num_labels <= 1:
        return 0
    
    # Skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.max(areas)) if len(areas) > 0 else 0
