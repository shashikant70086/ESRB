"""
Configuration module for stereo vision system.
Handles system settings, camera parameters, and calibration data.
"""

from .settings import (
    SystemConfig,
    CameraConfig,
    DetectionConfig,
    VisualizationConfig,
    AudioConfig,
    PerformanceConfig
)

__version__ = "1.0.0"
__all__ = [
    "SystemConfig",
    "CameraConfig", 
    "DetectionConfig",
    "VisualizationConfig",
    "AudioConfig",
    "PerformanceConfig"
]
