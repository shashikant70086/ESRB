# core/__init__.py
"""
Core processing modules for stereo vision system.
Contains camera management, stereo processing, detection, and tracking.
"""

from .camera_manager import CameraManager, ESP32CamStream
from .stereo_processor import StereoProcessor, DepthCalculator
from .detector_engine import DetectorEngine, MediaPipeDetector, YOLOv8Detector
from .tracker_system import TrackerSystem, ObjectTracker, TrackingState

__version__ = "1.0.0"
__all__ = [
    "CameraManager",
    "ESP32CamStream",
    "StereoProcessor", 
    "DepthCalculator",
    "DetectorEngine",
    "MediaPipeDetector",
    "YOLOv8Detector",
    "TrackerSystem",
    "ObjectTracker",
    "TrackingState"
]