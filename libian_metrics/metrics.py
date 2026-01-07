"""
Core metrics computation for ancient character glyph quality assessment.

Implements six key metrics:
    A1. SSI (Shape Squareness Index): External outline squareness
    A2. GCP (Global Centering of Mass): Center of mass proximity
    A3. SSD (Spatial Sparsity-Dispersion): Space distribution balance
    B1. STR (Straightness Ratio): Linear stroke proportion
    B2. CSI (Corner Sharpness Index): Corner sharpness intensity
    B3. COI (Connectivity & Overlap Index): Branch and loop complexity

And LQI (weighted composite score).
"""

import cv2
import numpy as np
from typing import Dict, Optional

from .skeleton import (
    extract_paths_from_skeleton,
    get_skeleton_angles,
    get_branch_points,
    count_endpoints
)


# Default weights for LQI computation
WEIGHTS = {
    'SSI': 0.20,
    'GCP': 0.10,
    'SSD': 0.25,
    'STR': 0.15,
    'CSI': 0.15,
    'COI': 0.15,
}


def ssi_squareness(bin_img: np.ndarray, r_cap: float = 3.0) -> float:
    """
    Shape Squareness Index: measures how square-shaped the outer contour is.
    
    Logic:
        - Find largest connected component
        - Get minimum area rotated rectangle (minAreaRect)
        - Compute aspect ratio r = max(w,h) / min(w,h)
        - Normalize: SSI = 1 - |log(r)| / |log(r_cap)|
        - Clamp to [0, 1]
    
    Args:
        bin_img: Binary image
        r_cap: Aspect ratio cap for normalization (default 3.0)
        
    Returns:
        SSI score in [0, 1]
    """
    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    
    if num_labels <= 1:
        return 0.0
    
    # Skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    
    # Get contour of largest component
    component_mask = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < 4:
        return 0.0
    
    # Get minimum area rotated rectangle
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    
    if min(w, h) < 1e-9:
        return 0.0
    
    # Aspect ratio
    r = max(w, h) / min(w, h)
    
    # Normalize with log scale
    ssi = 1.0 - abs(np.log(r)) / abs(np.log(r_cap))
    
    return float(np.clip(ssi, 0.0, 1.0))


def gcp_centering(bin_img: np.ndarray) -> float:
    """
    Global Centering of Mass: measures how centered the foreground is.
    
    Logic:
        - Get bounding box center (bx, by)
        - Compute foreground center of mass (cx, cy)
        - Distance normalized by diagonal
        - GCP = 1 - dist / diag
    
    Args:
        bin_img: Binary image
        
    Returns:
        GCP score in [0, 1]
    """
    # Get bounding box
    rows = np.any(bin_img, axis=1)
    cols = np.any(bin_img, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0.0
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Bounding box center
    bx = (x_min + x_max) / 2.0
    by = (y_min + y_max) / 2.0
    
    # Center of mass
    y_coords, x_coords = np.where(bin_img > 0)
    if len(x_coords) == 0:
        return 0.0
    
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    
    # Distance
    dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
    
    # Diagonal
    diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) + 1e-9
    
    gcp = 1.0 - (dist / diag)
    
    return float(np.clip(gcp, 0.0, 1.0))


def ssd_sparsity(bin_img: np.ndarray, grid: int = 5, c_cap: float = 1.0) -> float:
    """
    Spatial Sparsity-Dispersion: measures uniformity of foreground distribution.
    
    Logic:
        - Divide bounding box into grid x grid cells
        - Compute foreground pixel ratio in each cell
        - Calculate coefficient of variation CV = std(p) / mean(p)
        - Normalize: SSD = 1 - min(CV, c_cap) / c_cap
    
    Args:
        bin_img: Binary image
        grid: Grid resolution (default 5x5)
        c_cap: CV normalization cap (default 1.0)
        
    Returns:
        SSD score in [0, 1]
    """
    # Get bounding box
    rows = np.any(bin_img, axis=1)
    cols = np.any(bin_img, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0.0
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    
    # Extract region
    region = bin_img[y_min:y_max+1, x_min:x_max+1]
    region_pixels = region.sum()
    
    if region_pixels == 0:
        return 0.0
    
    # Divide into grid
    cell_h = h / grid
    cell_w = w / grid
    
    cell_ratios = []
    for i in range(grid):
        for j in range(grid):
            r1 = int(i * cell_h)
            r2 = int((i + 1) * cell_h)
            c1 = int(j * cell_w)
            c2 = int((j + 1) * cell_w)
            
            cell = region[r1:r2, c1:c2]
            cell_pixels = cell.sum()
            cell_area = cell.size
            
            ratio = (cell_pixels / (255 * cell_area + 1e-9)) if cell_area > 0 else 0.0
            cell_ratios.append(ratio)
    
    cell_ratios = np.array(cell_ratios)
    
    mean_ratio = np.mean(cell_ratios) + 1e-9
    std_ratio = np.std(cell_ratios)
    
    cv = std_ratio / mean_ratio
    
    ssd = 1.0 - min(cv, c_cap) / c_cap
    
    return float(np.clip(ssd, 0.0, 1.0))


def str_straightness(
    skel: np.ndarray,
    min_line_len_ratio: float = 0.08,
    max_line_gap: int = 3,
    use_main_angles: bool = True
) -> float:
    """
    Straightness Ratio: measures proportion of straight stroke segments.
    
    Logic:
        - Apply Probabilistic Hough on skeleton
        - Filter lines by angle (0/45/90/135° ± 10° if enabled)
        - Render lines on blank image, intersect with skeleton
        - STR = overlap_pixels / skel_pixels
    
    Args:
        skel: Skeleton image
        min_line_len_ratio: Minimum line length as fraction of image size
        max_line_gap: Maximum gap for line continuation
        use_main_angles: Filter to main angles (0/45/90/135°)
        
    Returns:
        STR score in [0, 1]
    """
    skel_pixels = cv2.countNonZero(skel)
    
    if skel_pixels < 10:
        return 0.0
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        skel.astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=max(5, int(max(skel.shape) * min_line_len_ratio)),
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return 0.0
    
    # Filter by angle if requested
    main_angles = [0, 45, 90, 135]
    angle_tolerance = 10
    
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Normalize angle to [0, 180)
        angle = angle % 180
        
        if use_main_angles:
            # Check if close to main angle
            is_main = False
            for ma in main_angles:
                if abs(angle - ma) < angle_tolerance or abs(angle - ma) > (180 - angle_tolerance):
                    is_main = True
                    break
            if is_main:
                filtered_lines.append(line[0])
        else:
            filtered_lines.append(line[0])
    
    if not filtered_lines:
        return 0.0
    
    # Render lines on blank image
    line_img = np.zeros_like(skel)
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
    
    # Intersect with skeleton
    overlap = cv2.bitwise_and(line_img, skel)
    overlap_pixels = cv2.countNonZero(overlap)
    
    str_score = overlap_pixels / (skel_pixels + 1e-9)
    
    return float(np.clip(str_score, 0.0, 1.0))


def csi_corner_sharpness(
    skel: np.ndarray,
    angle_thresh_deg: float = 30.0,
    density_alpha: float = 0.3
) -> float:
    """
    Corner Sharpness Index: measures sharpness of corners/junctions.
    
    Logic:
        - Extract skeleton paths
        - Calculate turning angles at each point
        - Identify corner points (|Δθ| > threshold)
        - CSI = (π - mean_angle)/π combined with corner density
    
    Args:
        skel: Skeleton image
        angle_thresh_deg: Threshold for corner detection in degrees
        density_alpha: Weight for density component
        
    Returns:
        CSI score in [0, 1]
    """
    angles = get_skeleton_angles(skel)
    
    if not angles:
        return 0.0
    
    angles = np.array(angles)
    
    # Identify corners: angles > threshold (less straight = more corner)
    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    corners = angles > angle_thresh_rad
    corner_count = np.sum(corners)
    
    if corner_count == 0:
        return 0.0
    
    # Mean angle for corner points
    mean_corner_angle = np.mean(angles[corners])
    
    # Sharpness from angle (π = 180° = most bent)
    angle_component = (np.pi - mean_corner_angle) / np.pi
    
    # Corner density
    skel_pixels = cv2.countNonZero(skel)
    density = corner_count / (skel_pixels + 1e-9) if skel_pixels > 0 else 0
    
    # Normalize density (empirical)
    density_normalized = min(density / 0.5, 1.0)  # Assume 0.5 is max expected
    
    # Combine
    csi = (1.0 - density_alpha) * angle_component + density_alpha * density_normalized
    
    return float(np.clip(csi, 0.0, 1.0))


def coi_connectivity_overlap(
    bin_img: np.ndarray,
    skel: np.ndarray,
    alpha: float = 0.6
) -> float:
    """
    Connectivity & Overlap Index: measures branch complexity and overlap regions.
    
    Logic:
        - Count branch points (degree >= 3) in skeleton
        - Estimate loops via morphological operations
        - COI = α * branch_density + (1-α) * loop_density
    
    Args:
        bin_img: Binary image
        skel: Skeleton image
        alpha: Weight for branch density component
        
    Returns:
        COI score in [0, 1]
    """
    skel_pixels = cv2.countNonZero(skel)
    
    if skel_pixels < 10:
        return 0.0
    
    # Branch point density
    branch_img = get_branch_points(skel)
    branch_count = cv2.countNonZero(branch_img)
    branch_density = branch_count / (skel_pixels + 1e-9)
    
    # Estimate loops/overlaps using morphological operations
    # Use opening to remove thin branches, then compare with original
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Calculate Euler number to estimate holes
    opened_bin = (opened > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened_bin, connectivity=8
    )
    num_components = num_labels - 1  # Exclude background
    
    # Simple loop estimation based on closure
    original_components, _, _, _ = cv2.connectedComponentsWithStats(
        (bin_img > 0).astype(np.uint8), connectivity=8
    )
    original_components -= 1  # Exclude background
    
    loop_estimate = max(0, original_components - num_components)
    
    # Normalize loop density
    max_loops = max(1, skel_pixels // 50)  # Empirical
    loop_density = min(loop_estimate / (max_loops + 1e-9), 1.0)
    
    # Normalize branch density
    max_branches = max(1, skel_pixels // 30)  # Empirical
    branch_density = min(branch_density / (max_branches / skel_pixels + 1e-9), 1.0)
    
    # Combine
    coi = alpha * branch_density + (1.0 - alpha) * loop_density
    
    return float(np.clip(coi, 0.0, 1.0))


def compute_all_metrics(
    bin_img: np.ndarray,
    skel: np.ndarray,
    calib: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all six metrics and weighted composite score (LQI).
    
    Args:
        bin_img: Binary image
        skel: Skeleton image
        calib: Calibration dictionary with optional parameters:
            - r_cap: SSI aspect ratio cap (default 3.0)
            - c_cap: SSD CV cap (default 1.0)
            - angle_thresh_deg: CSI angle threshold (default 30.0)
            
    Returns:
        Dictionary with keys: SSI, GCP, SSD, STR, CSI, COI, LQI
    """
    if calib is None:
        calib = {}
    
    # Compute individual metrics
    ssi = ssi_squareness(bin_img, r_cap=calib.get('r_cap', 3.0))
    gcp = gcp_centering(bin_img)
    ssd = ssd_sparsity(bin_img, grid=5, c_cap=calib.get('c_cap', 1.0))
    strr = str_straightness(skel, min_line_len_ratio=0.08, max_line_gap=3)
    csi = csi_corner_sharpness(skel, angle_thresh_deg=calib.get('angle_thresh_deg', 30.0))
    coi = coi_connectivity_overlap(bin_img, skel, alpha=0.6)
    
    # Compute weighted composite score
    lqi = (WEIGHTS['SSI'] * ssi +
           WEIGHTS['GCP'] * gcp +
           WEIGHTS['SSD'] * ssd +
           WEIGHTS['STR'] * strr +
           WEIGHTS['CSI'] * csi +
           WEIGHTS['COI'] * coi)
    
    return {
        'SSI': float(np.clip(ssi, 0.0, 1.0)),
        'GCP': float(np.clip(gcp, 0.0, 1.0)),
        'SSD': float(np.clip(ssd, 0.0, 1.0)),
        'STR': float(np.clip(strr, 0.0, 1.0)),
        'CSI': float(np.clip(csi, 0.0, 1.0)),
        'COI': float(np.clip(coi, 0.0, 1.0)),
        'LQI': float(np.clip(lqi, 0.0, 1.0)),
    }
