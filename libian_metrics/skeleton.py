"""
Skeleton extraction and processing utilities.

Functions:
    skeletonize: Extract skeleton from binary image
    prune_spurs: Remove short branches (spurs) from skeleton
    extract_paths_from_skeleton: Extract continuous paths from skeleton
    extract_segments: Extract line segments from skeleton
    get_branch_points: Find branching points in skeleton
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from scipy import ndimage


def skeletonize(bin_img: np.ndarray) -> np.ndarray:
    """
    Extract skeleton from binary image using scikit-image.
    
    Args:
        bin_img: Binary image (foreground=255)
        
    Returns:
        Skeleton image (0/255)
    """
    # Convert to bool (255 -> True)
    img_bool = bin_img > 127
    
    # Skeletonize
    skel_bool = sk_skeletonize(img_bool)
    
    # Convert back to 0/255
    skel = (skel_bool.astype(np.uint8) * 255)
    
    return skel


def prune_spurs(skel: np.ndarray, min_branch_len: int = 6) -> np.ndarray:
    """
    Remove short branches (spurs) from skeleton.
    
    Args:
        skel: Skeleton image (0/255)
        min_branch_len: Minimum branch length in pixels
        
    Returns:
        Pruned skeleton
    """
    result = skel.copy()
    
    # Iteratively remove endpoints with short branches
    for _ in range(50):  # Max iterations
        # Find endpoints (pixels with only 1 neighbor)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(result, kernel, iterations=1)
        endpoints = result - eroded
        
        if cv2.countNonZero(endpoints) == 0:
            break
        
        # Trace each endpoint backward
        endpoints_y, endpoints_x = np.where(endpoints > 0)
        
        for ep_y, ep_x in zip(endpoints_y, endpoints_x):
            # Trace backward from endpoint
            branch_pixels = [(ep_y, ep_x)]
            current = (ep_y, ep_x)
            visited = {current}
            
            for _ in range(min_branch_len + 1):
                # Find next neighbor
                y, x = current
                found_next = False
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        
                        if (0 <= ny < result.shape[0] and 
                            0 <= nx < result.shape[1] and
                            result[ny, nx] > 0 and
                            (ny, nx) not in visited):
                            
                            branch_pixels.append((ny, nx))
                            visited.add((ny, nx))
                            current = (ny, nx)
                            found_next = True
                            break
                    
                    if found_next:
                        break
                
                if not found_next:
                    break
            
            # If branch is shorter than threshold, remove it
            if len(branch_pixels) <= min_branch_len:
                for py, px in branch_pixels:
                    result[py, px] = 0
    
    return result


def extract_paths_from_skeleton(skel: np.ndarray) -> list[np.ndarray]:
    """
    Extract continuous paths (chains of connected pixels) from skeleton.
    
    Args:
        skel: Skeleton image (0/255)
        
    Returns:
        List of paths, each path is array of shape (N, 2) with (y, x) coordinates
    """
    skel_bool = skel > 127
    visited = np.zeros_like(skel_bool)
    paths = []
    
    # Find all skeleton pixels
    skel_points = np.where(skel_bool)
    
    for start_y, start_x in zip(skel_points[0], skel_points[1]):
        if visited[start_y, start_x]:
            continue
        
        # Trace path from this starting point
        path = []
        current = (start_y, start_x)
        prev = None
        
        while True:
            y, x = current
            
            if visited[y, x]:
                break
            
            visited[y, x] = True
            path.append([y, x])
            
            # Find next unvisited neighbor
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < skel.shape[0] and
                        0 <= nx < skel.shape[1] and
                        skel_bool[ny, nx] and
                        not visited[ny, nx]):
                        
                        current = (ny, nx)
                        found_next = True
                        break
                
                if found_next:
                    break
            
            if not found_next:
                break
        
        if len(path) > 1:  # Only keep paths with at least 2 pixels
            paths.append(np.array(path, dtype=np.float32))
    
    return paths


def extract_segments(skel: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Extract line segments from skeleton using Hough line detection.
    
    Args:
        skel: Skeleton image (0/255)
        
    Returns:
        List of segments as (x1, y1, x2, y2) tuples
    """
    lines = cv2.HoughLinesP(
        skel.astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=5,
        maxLineGap=2
    )
    
    segments = []
    if lines is not None:
        for line in lines:
            segments.append(tuple(line[0]))
    
    return segments


def get_branch_points(skel: np.ndarray) -> np.ndarray:
    """
    Find branching points (degree >= 3) in skeleton.
    
    Args:
        skel: Skeleton image (0/255)
        
    Returns:
        Binary image with branching points marked
    """
    skel_bool = skel > 127
    
    # Use morphological convolution to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
    
    # Branch points have >= 3 neighbors
    branch_points = np.zeros_like(skel_bool)
    branch_points[(neighbor_count >= 3) & skel_bool] = True
    
    return (branch_points.astype(np.uint8) * 255)


def get_skeleton_angles(skel: np.ndarray) -> list[float]:
    """
    Extract angle information from skeleton paths.
    
    Args:
        skel: Skeleton image (0/255)
        
    Returns:
        List of direction angles (in radians) along skeleton
    """
    paths = extract_paths_from_skeleton(skel)
    angles = []
    
    for path in paths:
        if len(path) < 2:
            continue
        
        # Calculate angle changes along the path
        for i in range(1, len(path) - 1):
            p_prev = path[i - 1]
            p_curr = path[i]
            p_next = path[i + 1]
            
            # Vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Normalize
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Compute angle
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(float(angle))
    
    return angles


def count_endpoints(skel: np.ndarray) -> int:
    """
    Count endpoints in skeleton (degree = 1).
    
    Args:
        skel: Skeleton image (0/255)
        
    Returns:
        Number of endpoints
    """
    skel_bool = skel > 127
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
    
    # Endpoints have exactly 1 neighbor
    endpoints = np.sum((neighbor_count == 1) & skel_bool)
    
    return int(endpoints)
