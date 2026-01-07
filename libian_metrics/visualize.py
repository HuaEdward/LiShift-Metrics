"""
Visualization module for metrics and intermediate results.

Functions:
    plot_metrics: Create bar chart of all metrics with LQI highlight
    plot_preprocessing_stages: Visualize preprocessing pipeline stages
    save_metric_visualization: Save metrics visualization to file
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Optional
import cv2


def plot_metrics(metrics: Dict[str, float], output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Create a visualization of all metrics with LQI highlighted.
    
    Args:
        metrics: Dictionary with metric names and scores
        output_path: Optional path to save the figure
        
    Returns:
        None if saved to file, otherwise the figure as numpy array
    """
    # Extract metrics (exclude metadata)
    metric_names = ['SSI', 'GCP', 'SSD', 'STR', 'CSI', 'COI', 'LQI']
    metric_values = [metrics.get(name, 0.0) for name in metric_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors - LQI in different color
    colors = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
    
    # Create bar chart
    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize axes
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Ancient Character Glyph Quality Metrics\n(Layout Metrics: Blue | Stroke Metrics: Red | Composite: Green)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add weight annotations below names
    weights_text = ['0.20', '0.10', '0.25', '0.15', '0.15', '0.15', 'LQI']
    for i, (name, weight) in enumerate(zip(metric_names, weights_text)):
        ax.text(i, -0.08, f'w={weight}', ha='center', va='top', fontsize=9,
                transform=ax.get_xaxis_transform(), style='italic', color='gray')
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.text(-0.5, 0.5, '0.50', fontsize=9, color='gray', alpha=0.7)
    ax.text(-0.5, 0.7, '0.70', fontsize=9, color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Metrics visualization saved to {output_path}")
        plt.close()
        return None
    else:
        # Return as array if not saving to file
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data


def plot_preprocessing_stages(
    original: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    skeleton: np.ndarray,
    pruned_skel: np.ndarray,
    output_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Visualize preprocessing pipeline stages.
    
    Args:
        original: Original BGR image
        gray: Grayscale image
        binary: Binary image
        skeleton: Skeleton image
        pruned_skel: Pruned skeleton
        output_path: Optional path to save the figure
        
    Returns:
        None if saved to file
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Stage 1: Original
    if len(original.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Stage 2: Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('2. Grayscale', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Stage 3: Binary
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('3. Adaptive Binary', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Stage 4: Skeleton
    axes[1, 0].imshow(skeleton, cmap='gray')
    axes[1, 0].set_title('4. Skeleton', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Stage 5: Pruned Skeleton
    axes[1, 1].imshow(pruned_skel, cmap='gray')
    axes[1, 1].set_title('5. Pruned Skeleton', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Stage 6: Overlay
    overlay = pruned_skel.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('6. Final Result', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Preprocessing Pipeline Stages', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Preprocessing visualization saved to {output_path}")
        plt.close()
        return None
    else:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data


def create_metric_summary_image(
    metrics: Dict[str, float],
    image_path: str,
    output_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Create a summary image with metrics and metadata.
    
    Args:
        metrics: Dictionary with all metrics and metadata
        image_path: Path to the processed image
        output_path: Optional path to save the figure
        
    Returns:
        None if saved to file
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Create text-based summary
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    title_text = f"LiBian Metrics Summary\n{os.path.basename(image_path)}"
    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=14, 
            fontweight='bold', transform=ax.transAxes)
    
    # Metrics section
    y_start = 0.85
    
    # Layout metrics
    layout_text = "Layout Metrics (Weight: 0.55):\n"
    layout_text += f"  • SSI (Squareness):        {metrics.get('SSI', 0.0):.4f}  [0.20]\n"
    layout_text += f"  • GCP (Centering):         {metrics.get('GCP', 0.0):.4f}  [0.10]\n"
    layout_text += f"  • SSD (Sparsity):          {metrics.get('SSD', 0.0):.4f}  [0.25]\n"
    
    ax.text(0.05, y_start, layout_text, ha='left', va='top', fontsize=11,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Stroke metrics
    y_start -= 0.25
    stroke_text = "Stroke Metrics (Weight: 0.45):\n"
    stroke_text += f"  • STR (Straightness):      {metrics.get('STR', 0.0):.4f}  [0.15]\n"
    stroke_text += f"  • CSI (Corner Sharpness):  {metrics.get('CSI', 0.0):.4f}  [0.15]\n"
    stroke_text += f"  • COI (Connectivity):      {metrics.get('COI', 0.0):.4f}  [0.15]\n"
    
    ax.text(0.05, y_start, stroke_text, ha='left', va='top', fontsize=11,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # LQI
    y_start -= 0.25
    lqi_value = metrics.get('LQI', 0.0)
    lqi_color = 'lightgreen' if lqi_value >= 0.7 else 'lightyellow' if lqi_value >= 0.5 else 'lightcoral'
    lqi_text = f"LQI (Composite Score): {lqi_value:.4f}"
    ax.text(0.05, y_start, lqi_text, ha='left', va='top', fontsize=13,
            fontweight='bold', family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=lqi_color, alpha=0.5, pad=0.8))
    
    # Metadata
    y_start -= 0.15
    meta_text = f"Image Processing Info:\n"
    meta_text += f"  • Quality Flag: {metrics.get('quality_flag', False)}\n"
    meta_text += f"  • Rotation Angle: {metrics.get('angle', 0.0):.2f}°\n"
    meta_text += f"  • Scale Factor: {metrics.get('scale', 1.0):.4f}\n"
    meta_text += f"  • Skeleton Pixels: {metrics.get('skel_pixels', 0)}\n"
    
    ax.text(0.05, y_start, meta_text, ha='left', va='top', fontsize=10,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Summary visualization saved to {output_path}")
        plt.close()
        return None
    else:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data


def save_metric_visualization(
    metrics: Dict[str, float],
    image_path: str,
    output_dir: str
) -> Tuple[str, str]:
    """
    Save all visualizations for an image analysis.
    
    Args:
        metrics: Dictionary with all metrics
        image_path: Path to the original image
        output_dir: Directory to save visualizations
        
    Returns:
        (metrics_chart_path, summary_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    metrics_chart_path = os.path.join(output_dir, f'{base_name}_metrics.png')
    summary_path = os.path.join(output_dir, f'{base_name}_summary.png')
    
    # Save visualizations
    plot_metrics(metrics, output_path=metrics_chart_path)
    create_metric_summary_image(metrics, image_path, output_path=summary_path)
    
    return metrics_chart_path, summary_path
