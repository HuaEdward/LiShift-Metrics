"""
LiBian Metrics: Quantitative evaluation tool for ancient character glyphs.

This package provides comprehensive metrics for assessing the quality of 
ancient character (glyph) images, including:
- Layout metrics (SSI, GCP, SSD)
- Stroke metrics (STR, CSI, COI)
- Weighted composite score (LQI)
"""

__version__ = "0.1.0"
__author__ = "LiShift Team"

from .preprocess import preprocess
from .metrics import compute_all_metrics

__all__ = ["preprocess", "compute_all_metrics"]
