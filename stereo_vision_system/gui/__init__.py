# gui/__init__.py
"""
GUI module for stereo vision system.
Provides PyQt6-based control panel and user interface.
"""

from .control_panel import (
    ControlPanel,
    CameraControlWidget,
    DetectionControlWidget,
    VisualizationControlWidget,
    SystemStatusWidget
)

__version__ = "1.0.0"
__all__ = [
    "ControlPanel",
    "CameraControlWidget",
    "DetectionControlWidget", 
    "VisualizationControlWidget",
    "SystemStatusWidget"
]
