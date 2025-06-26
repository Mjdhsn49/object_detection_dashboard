"""
Model implementations for object detection and depth estimation.
"""

from .detection_model import ObjectDetector
from .depth_model import DepthEstimator

__all__ = ['ObjectDetector', 'DepthEstimator'] 