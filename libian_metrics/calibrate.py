"""
Calibration module for parameter tuning based on sample images.

Functions:
    calibrate_from_folder: Generate calibration.json from image samples
    load_calibration: Load calibration from JSON file
    save_calibration: Save calibration to JSON file
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from .io_utils import read_image, remove_small_components
from .preprocess import preprocess
from .skeleton import (
    extract_paths_from_skeleton,
    get_skeleton_angles,
    get_branch_points
)
import cv2


def calibrate_from_folder(
    folder: str,
    sample_n: Optional[int] = None,
    r_cap_percentile: float = 99.0,
    c_cap_percentile: float = 95.0
) -> Dict:
    """
    Generate calibration parameters from a folder of sample images.
    
    Args:
        folder: Path to folder containing image samples
        sample_n: Maximum number of samples to use (None = use all)
        r_cap_percentile: Percentile for aspect ratio cap
        c_cap_percentile: Percentile for CV normalization cap
        
    Returns:
        Calibration dictionary with parameters:
        - r_cap: Aspect ratio cap
        - c_cap: CV normalization cap
        - angle_thresh_deg: Corner detection threshold
        - density_alpha: Branch density weight
    """
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {folder}")
        return _get_default_calibration()
    
    # Limit sample count
    if sample_n is not None:
        image_files = image_files[:sample_n]
    
    print(f"Calibrating with {len(image_files)} samples...")
    
    r_values = []
    cv_values = []
    corner_densities = []
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"  Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            img = read_image(img_path)
            bin_img, skel, meta = preprocess(img)
            
            # Extract metrics for calibration
            
            # 1. Aspect ratio distribution (SSI calibration)
            r = _get_aspect_ratio(bin_img)
            if r is not None:
                r_values.append(r)
            
            # 2. CV distribution (SSD calibration)
            cv = _get_spatial_cv(bin_img)
            if cv is not None:
                cv_values.append(cv)
            
            # 3. Corner density (CSI calibration)
            density = _get_corner_density(skel)
            if density is not None:
                corner_densities.append(density)
        
        except Exception as e:
            print(f"    Error processing {img_path}: {e}")
            continue
    
    # Compute percentiles
    if r_values:
        r_cap = np.percentile(r_values, r_cap_percentile)
    else:
        r_cap = 3.0
    
    if cv_values:
        c_cap = np.percentile(cv_values, c_cap_percentile)
    else:
        c_cap = 1.0
    
    if corner_densities:
        angle_thresh_deg = np.percentile(corner_densities, 50) * 100  # Empirical
    else:
        angle_thresh_deg = 30.0
    
    calib = {
        'r_cap': float(np.clip(r_cap, 1.5, 10.0)),
        'c_cap': float(np.clip(c_cap, 0.5, 5.0)),
        'angle_thresh_deg': float(np.clip(angle_thresh_deg, 10.0, 60.0)),
        'density_alpha': 0.6,
        'num_samples': len(image_files),
        'r_values_percentiles': {
            'min': float(np.min(r_values)) if r_values else 1.0,
            'p25': float(np.percentile(r_values, 25)) if r_values else 1.0,
            'p50': float(np.percentile(r_values, 50)) if r_values else 1.0,
            'p75': float(np.percentile(r_values, 75)) if r_values else 1.0,
            'max': float(np.max(r_values)) if r_values else 1.0,
        },
        'cv_values_percentiles': {
            'min': float(np.min(cv_values)) if cv_values else 0.0,
            'p25': float(np.percentile(cv_values, 25)) if cv_values else 0.0,
            'p50': float(np.percentile(cv_values, 50)) if cv_values else 0.0,
            'p75': float(np.percentile(cv_values, 75)) if cv_values else 0.0,
            'max': float(np.max(cv_values)) if cv_values else 0.0,
        },
    }
    
    print(f"Calibration complete:")
    print(f"  r_cap = {calib['r_cap']:.2f}")
    print(f"  c_cap = {calib['c_cap']:.2f}")
    print(f"  angle_thresh_deg = {calib['angle_thresh_deg']:.2f}")
    
    return calib


def _get_aspect_ratio(bin_img: np.ndarray) -> Optional[float]:
    """Get aspect ratio of largest component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    
    if num_labels <= 1:
        return None
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    
    component_mask = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return None
    
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    
    if min(w, h) < 1e-9:
        return None
    
    r = max(w, h) / min(w, h)
    return float(r)


def _get_spatial_cv(bin_img: np.ndarray, grid: int = 5) -> Optional[float]:
    """Get coefficient of variation for spatial distribution."""
    rows = np.any(bin_img, axis=1)
    cols = np.any(bin_img, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    
    region = bin_img[y_min:y_max+1, x_min:x_max+1]
    region_pixels = region.sum()
    
    if region_pixels == 0:
        return None
    
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
    return float(cv)


def _get_corner_density(skel: np.ndarray) -> Optional[float]:
    """Get corner point density in skeleton."""
    angles = get_skeleton_angles(skel)
    if not angles:
        return None
    
    angles = np.array(angles)
    skel_pixels = cv2.countNonZero(skel)
    
    if skel_pixels < 10:
        return None
    
    corner_thresh = np.deg2rad(30)
    corners = angles > corner_thresh
    corner_count = np.sum(corners)
    
    density = corner_count / skel_pixels
    return float(density)


def load_calibration(calib_path: str) -> Dict:
    """
    Load calibration from JSON file.
    
    Args:
        calib_path: Path to calibration JSON file
        
    Returns:
        Calibration dictionary
    """
    try:
        with open(calib_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load calibration from {calib_path}: {e}")
        return _get_default_calibration()


def save_calibration(calib: Dict, output_path: str) -> None:
    """
    Save calibration to JSON file.
    
    Args:
        calib: Calibration dictionary
        output_path: Output JSON file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"Calibration saved to {output_path}")


def _get_default_calibration() -> Dict:
    """Get default calibration parameters."""
    return {
        'r_cap': 3.0,
        'c_cap': 1.0,
        'angle_thresh_deg': 30.0,
        'density_alpha': 0.6,
    }
