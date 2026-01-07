"""
I/O utilities for image loading and basic preprocessing.

Functions:
    read_image: Load image from file (support JPG, PNG)
    remove_small_components: Remove small connected components from binary image
    estimate_skew_angle: Estimate image rotation angle using Hough line detection
    rotate_bound: Rotate image and adjust bounds
"""

import cv2
import numpy as np
from scipy import ndimage


def read_image(image_path: str) -> np.ndarray:
    """
    Load image from file (JPG/PNG).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image in BGR format (numpy array)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return img


def remove_small_components(bin_img: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove small connected components from binary image.
    
    Args:
        bin_img: Binary image (foreground=255, background=0)
        min_area: Minimum pixel count for a component to be kept
        
    Returns:
        Cleaned binary image with small components removed
    """
    result = bin_img.copy()
    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    
    # Remove components smaller than min_area (skip background label 0)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            result[labels == label] = 0
    
    return result


def estimate_skew_angle(bin_img: np.ndarray, angle_range: float = 5.0) -> float:
    """
    Estimate image skew angle using probabilistic Hough line detection.
    
    Args:
        bin_img: Binary image (foreground=255)
        angle_range: Acceptable skew range in degrees (±angle_range)
        
    Returns:
        Rotation angle in degrees (within ±angle_range). 
        Positive angle = counterclockwise rotation.
    """
    # Use Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        bin_img,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Extract angles from lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Normalize angle to [-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Find dominant angle using histogram
    angles_arr = np.array(angles)
    hist, bin_edges = np.histogram(angles_arr, bins=30, range=(-90, 90))
    dominant_idx = np.argmax(hist)
    dominant_angle = (bin_edges[dominant_idx] + bin_edges[dominant_idx + 1]) / 2
    
    # Clamp to ±angle_range
    dominant_angle = np.clip(dominant_angle, -angle_range, angle_range)
    
    return float(dominant_angle)


def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by angle and adjust bounds to include full image.
    
    Args:
        img: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image with white padding
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounds
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation with white padding
    rotated = cv2.warpAffine(
        img, rot_mat, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(img.shape) == 3 else 255
    )
    
    return rotated


def crop_to_content(bin_img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop binary image to bounding box of foreground (non-zero pixels).
    
    Args:
        bin_img: Binary image
        
    Returns:
        (cropped_image, bbox) where bbox = (y0, y1, x0, x1)
    """
    rows = np.any(bin_img, axis=1)
    cols = np.any(bin_img, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty image
        return bin_img, (0, bin_img.shape[0], 0, bin_img.shape[1])
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    cropped = bin_img[y_min:y_max+1, x_min:x_max+1]
    bbox = (y_min, y_max + 1, x_min, x_max + 1)
    
    return cropped, bbox


def pad_to_size(img: np.ndarray, target_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Pad image to square with given size, maintaining aspect ratio.
    
    Args:
        img: Input image
        target_size: Target size (both width and height)
        
    Returns:
        (padded_image, (pad_h, pad_w)) - padding applied
    """
    h, w = img.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    # Pad evenly on both sides
    pad_h_top = pad_h // 2
    pad_h_bot = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    if len(img.shape) == 2:
        padded = cv2.copyMakeBorder(
            img,
            pad_h_top, pad_h_bot, pad_w_left, pad_w_right,
            cv2.BORDER_CONSTANT, value=255
        )
    else:
        padded = cv2.copyMakeBorder(
            img,
            pad_h_top, pad_h_bot, pad_w_left, pad_w_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    
    return padded, (pad_h, pad_w)
