# Root __init__.py for stereo_vision_system package
"""
Real-time Stereo Vision System with AI Object Detection
A production-grade system for dual ESP32-CAM stereo vision with distance estimation.

Features:
- Dual ESP32-CAM IP stream capture
- Real-time stereo vision depth calculation  
- AI-powered object detection (MediaPipe + YOLOv8)
- Multi-object tracking with unique IDs
- Futuristic HUD overlay with distance annotations
- 2D radar mapping and optional 3D visualization
- Audio feedback system
- PyQt6 GUI control panel
- GPU-accelerated inference on RTX 3050
"""

from . import config
from . import core
from . import visualization
from . import audio
from . import gui
from . import utils

__version__ = "1.0.0"
__author__ = "Stereo Vision Team"
__description__ = "Real-time Stereo Vision System with AI Object Detection"

__all__ = [
    "config",
    "core", 
    "visualization",
    "audio",
    "gui",
    "utils"
]

# System information
SYSTEM_INFO = {
    "name": "Stereo Vision System",
    "version": __version__,
    "python_required": "3.11+",
    "gpu_support": ["NVIDIA RTX 3050", "CUDA"],
    "camera_support": ["ESP32-CAM"],
    "ai_frameworks": ["MediaPipe", "YOLOv8", "TensorRT"],
    "gui_framework": "PyQt6"
}