# config/settings.py
"""
Core configuration settings for the stereo vision system
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    baseline_cm: float = 10.0  # Distance between cameras in cm
    left_ip: str = "192.168.1.100"
    right_ip: str = "192.168.1.101"
    left_rotation: int = 0  # 0, 90, 180, 270 degrees
    right_rotation: int = 0
    left_mirror: bool = False
    right_mirror: bool = False
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 15
    focal_length_px: float = 500.0  # Will be calibrated
    
@dataclass
class DetectionConfig:
    """AI detection configuration"""
    enable_mediapipe: bool = True
    enable_yolo: bool = True
    yolo_model: str = "yolov8n.pt"  # nano for speed
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 10
    use_tensorrt: bool = True
    device: str = "cuda:0"  # RTX 3050
    
@dataclass
class StereoConfig:
    """Stereo vision parameters"""
    stereo_algorithm: str = "SGBM"  # "BM" or "SGBM"
    num_disparities: int = 64
    block_size: int = 11
    min_disparity: int = 0
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    disp12_max_diff: int = 1
    
@dataclass
class HUDConfig:
    """HUD visualization settings"""
    enable_hud: bool = True
    show_fps: bool = True
    show_distance: bool = True
    show_scan_lines: bool = True
    hud_color_normal: Tuple[int, int, int] = (0, 255, 0)  # Green
    hud_color_warning: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    hud_color_danger: Tuple[int, int, int] = (0, 0, 255)  # Red
    danger_distance_m: float = 2.0
    warning_distance_m: float = 5.0
    
@dataclass
class AudioConfig:
    """Audio feedback configuration"""
    enable_tts: bool = True
    tts_engine: str = "pyttsx3"  # or "bark"
    voice_rate: int = 150
    voice_volume: float = 0.9
    announce_new_detection: bool = True
    announce_distance: bool = True
    min_announce_interval: float = 3.0  # seconds
    
@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    use_threading: bool = True
    max_threads: int = 4
    frame_buffer_size: int = 5
    skip_frames: int = 1  # Process every nth frame
    enable_gpu_acceleration: bool = True
    max_fps: int = 20
    
@dataclass
class SystemConfig:
    """Main system configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # System paths
    config_dir: str = "config"
    models_dir: str = "models"
    logs_dir: str = "logs"
    snapshots_dir: str = "snapshots"
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

# Global configuration instance
CONFIG = SystemConfig()

# Camera calibration matrix template
CAMERA_MATRIX_TEMPLATE = {
    "left_camera": {
        "camera_matrix": [[500.0, 0.0, 320.0], 
                         [0.0, 500.0, 240.0], 
                         [0.0, 0.0, 1.0]],
        "distortion_coeffs": [0.1, -0.2, 0.0, 0.0, 0.0]
    },
    "right_camera": {
        "camera_matrix": [[500.0, 0.0, 320.0], 
                         [0.0, 500.0, 240.0], 
                         [0.0, 0.0, 1.0]],
        "distortion_coeffs": [0.1, -0.2, 0.0, 0.0, 0.0]
    },
    "stereo_params": {
        "rotation_matrix": [[1.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0], 
                           [0.0, 0.0, 1.0]],
        "translation_vector": [-10.0, 0.0, 0.0],  # baseline in cm
        "essential_matrix": [],
        "fundamental_matrix": []
    }
}

def create_default_config():
    """Create default configuration files"""
    os.makedirs("config", exist_ok=True)
    
    # Save default system config
    CONFIG.save_to_file("config/system_config.json")
    
    # Save camera calibration template
    with open("config/camera_config.json", 'w') as f:
        json.dump(CAMERA_MATRIX_TEMPLATE, f, indent=4)
    
    print("✓ Default configuration files created")
    
if __name__ == "__main__":
    create_default_config()