# main.py
import cv2
import numpy as np
import time
import threading
import argparse
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import SystemConfig
from core.camera_manager import CameraManager
from core.stereo_processor import StereoProcessor
from core.detector_engine import DetectorEngine
from core.tracker_system import MultiObjectTracker
from visualization.hud_overlay import HUDOverlay
from visualization.radar_mapper import RadarMapper
from audio.tts_engine import TTSEngine
from utils.performance import PerformanceMonitor
from utils.logger import SystemLogger

class StereoVisionSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = SystemLogger()
        self.performance = PerformanceMonitor()
        
        # Core components
        self.camera_manager = None
        self.stereo_processor = None
        self.detector_engine = None
        self.tracker = None
        self.hud_overlay = None
        self.radar_mapper = None
        self.tts_engine = None
        
        # System state
        self.is_running = False
        self.frame_count = 0
        self.last_status_time = time.time()
        
        # Threading
        self.processing_lock = threading.Lock()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Stereo Vision System...")
            
            # Initialize camera manager
            self.logger.info("Setting up cameras...")
            self.camera_manager = CameraManager(
                left_ip=self.config.left_camera_ip,
                right_ip=self.config.right_camera_ip,
                left_rotation=self.config.left_camera_rotation,
                right_rotation=self.config.right_camera_rotation
            )
            
            if not self.camera_manager.initialize():
                self.logger.error("Failed to initialize cameras")
                return False
            
            # Get frame dimensions
            frame_width, frame_height = self.camera_manager.get_frame_size()
            self.logger.info(f"Camera resolution: {frame_width}x{frame_height}")
            
            # Initialize stereo processor
            self.logger.info("Setting up stereo vision...")
            self.stereo_processor = StereoProcessor(
                baseline_cm=self.config.baseline_distance,
                frame_width=frame_width,
                frame_height=frame_height
            )
            
            # Initialize AI detector
            self.logger.info("Loading AI models...")
            self.detector_engine = DetectorEngine(
                use_mediapipe=self.config.use_mediapipe,
                use_yolo=self.config.use_yolo,
                confidence_threshold=self.config.detection_confidence
            )
            
            if not self.detector_engine.initialize():
                self.logger.error("Failed to initialize AI detector")
                return False
            
            # Initialize tracker
            self.logger.info("Setting up object tracking...")
            self.tracker = MultiObjectTracker(
                max_disappeared=self.config.tracking_max_disappeared,
                max_distance=self.config.tracking_max_distance
            )
            
            # Initialize visualization
            self.logger.info("Setting up visualization...")
            self.hud_overlay = HUDOverlay(frame_width, frame_height)
            self.radar_mapper = RadarMapper(
                radar_size=self.config.radar_size,
                max_range=self.config.radar_max_range
            )
            
            # Initialize audio system
            if self.config.enable_audio:
                self.logger.info("Setting up audio system...")
                self.tts_engine = TTSEngine()
                self.tts_engine.set_voice_settings(
                    rate=self.config.tts_rate,
                    volume=self.config.tts_volume
                )
            
            self.logger.info("System initialization complete!")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def process_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a pair of stereo frames"""
        with self.processing_lock:
            # Generate disparity map
            disparity = self.stereo_processor.compute_disparity(left_frame, right_frame)
            
            # Detect objects in left frame
            detections = self.detector_engine.detect_objects(left_frame)
            
            # Calculate distances for detections
            for detection in detections:
                bbox = detection['bbox']
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                
                # Get distance from disparity map
                distance = self.stereo_processor.get_distance_at_point(disparity, center_x, center_y)
                detection['distance'] = distance
            
            # Update tracker
            tracked_objects = self.tracker.update(left_frame, detections)
            
            # Audio announcements
            if self.tts_engine and tracked_objects:
                for obj in tracked_objects:
                    # Announce new detections
                    if obj.id not in [existing.id for existing in self.tracker.objects.values() if existing.id != obj.id]:
                        self.tts_engine.announce_detection(obj)
                
                # Periodic status announcement
                current_time = time.time()
                if current_time - self.last_status_time > self.config.status_announcement_interval:
                    self.tts_engine.announce_status(tracked_objects)
                    self.last_status_time = current_time
            
            # Generate visualizations
            hud_frame = self.hud_overlay.render(left_frame, tracked_objects)
            radar_frame = self.radar_mapper.render(tracked_objects, left_frame.shape[1], left_frame.shape[0])
            
            return hud_frame, radar_frame
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input during operation"""
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            return False
        elif key == ord('s'):  # 's' to save snapshot
            self.save_snapshot()
        elif key == ord('c'):  # 'c' to clear tracker
            self.tracker.clear()
            if self.radar_mapper:
                self.radar_mapper.clear_trails()
            self.logger.info("Tracker and trails cleared")
        elif key == ord('a'):  # 'a' to toggle audio
            if self.tts_engine:
                self.tts_engine.announce_custom("Audio system active")
        elif key == ord('r'):  # 'r' to reset HUD
            if self.hud_overlay:
                self.hud_overlay.reset_frame_count()
        elif key == ord('h'):  # 'h' for help
            self.print_help()
        
        return True
    
    def save_snapshot(self):
        """Save current frame snapshot"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Get current frames
            left_frame, right_frame = self.camera_manager.get_frame_pair()
            if left_frame is not None and right_frame is not None:
                # Save left frame with HUD
                hud_frame, _ = self.process_frame_pair(left_frame, right_frame)
                filename = f"snapshot_{timestamp}_hud.jpg"
                cv2.imwrite(filename, hud_frame)
                
                # Save raw stereo pair
                combined = np.hstack([left_frame, right_frame])
                raw_filename = f"snapshot_{timestamp}_stereo.jpg"
                cv2.imwrite(raw_filename, combined)
                
                self.logger.info(f"Snapshots saved: {filename}, {raw_filename}")
                
                if self.tts_engine:
                    self.tts_engine.announce_custom("Snapshot saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
    
    def print_help(self):
        """Print keyboard controls help"""
        help_text = """
        Keyboard Controls:
        Q / ESC - Quit application
        S - Save snapshot
        C - Clear tracker and trails
        A - Audio test
        R - Reset HUD counters
        H - Show this help
        """
        print(help_text)
    
    def run(self):
        """Main system execution loop"""
        if not self.initialize_system():
            self.logger.error("System initialization failed. Exiting.")
            return