#!/usr/bin/env python3
"""
Production Stereo Vision System
Real-time stereo vision with AI detection, tracking, and advanced logging
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time
import signal
import json
from threading import Thread, Event
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import our modules
from config.settings import SystemConfig
from core.camera_manager import DualCameraManager
from core.stereo_processor import StereoVisionProcessor
from core.detector_engine import DetectionEngine
from core.tracker_system import ObjectTracker
from visualization.hud_overlay import HUDOverlay
from visualization.radar_mapper import RadarMapper
from visualization.point_cloud import PointCloudVisualizer
from audio.tts_engine import TTSEngine
from utils.performance import PerformanceMonitor, AdaptiveFrameSkipper, ResourceOptimizer
from utils.logger import (
    init_logger, get_logger, log_performance, log_method_calls,
    log_info, log_warning, log_error, log_debug
)

class StereoVisionSystem:
    """Main stereo vision system coordinator with advanced logging and error handling"""
    
    def __init__(self, config: SystemConfig, gui_mode: bool = False):
        self.config = config
        self.gui_mode = gui_mode
        self.running = False
        self.shutdown_event = Event()
        self.system_start_time = time.time()
        
        # Get logger instance
        self.logger = get_logger()
        self.system_logger = self.logger.get_component_logger('StereoVisionSystem')
        
        # Initialize components to None
        self.camera_manager = None
        self.stereo_processor = None
        self.detector = None
        self.tracker = None
        self.hud = None
        self.radar = None
        self.point_cloud = None
        self.tts = None
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        self.frame_skipper = AdaptiveFrameSkipper(target_fps=config.target_fps)
        self.resource_optimizer = ResourceOptimizer()
        
        # GUI components
        self.gui_app = None
        self.gui_window = None
        
        # Processing thread
        self.processing_thread = None
        
        # System state tracking
        self.total_frames_processed = 0
        self.total_detections = 0
        self.last_heartbeat = time.time()
        
        # Error recovery
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        
        # Initialize system
        self._initialize_components()
        
        # Log system configuration
        self.logger.log_camera_config({
            'baseline_distance': config.baseline_distance,
            'left_camera_url': config.left_camera_url,
            'right_camera_url': config.right_camera_url,
            'target_fps': config.target_fps,
            'confidence_threshold': config.confidence_threshold
        })
    
    @log_method_calls(get_logger(), 'StereoVisionSystem')
    def _initialize_components(self):
        """Initialize all system components with error handling"""
        try:
            self.system_logger.info("Initializing stereo vision system...")
            
            # Log system information
            self.logger.log_system_info()
            
            # Optimize system resources
            self.system_logger.info("Optimizing system resources...")
            self.resource_optimizer.set_high_priority()
            self.resource_optimizer.optimize_opencv_threads()
            
            # Initialize camera manager
            self.system_logger.info("Initializing camera manager...")
            self.camera_manager = DualCameraManager(
                left_url=self.config.left_camera_url,
                right_url=self.config.right_camera_url,
                left_rotation=self.config.left_rotation,
                right_rotation=self.config.right_rotation
            )
            
            # Initialize stereo processor
            self.system_logger.info("Initializing stereo processor...")
            self.stereo_processor = StereoVisionProcessor(
                baseline_cm=self.config.baseline_distance,
                focal_length=self.config.focal_length
            )
            
            # Initialize detection engine
            self.system_logger.info("Initializing AI detection engine...")
            self.detector = DetectionEngine(
                use_mediapipe=self.config.use_mediapipe,
                use_yolo=self.config.use_yolo,
                confidence_threshold=self.config.confidence_threshold
            )
            
            # Initialize tracker
            self.system_logger.info("Initializing object tracker...")
            self.tracker = ObjectTracker()
            
            # Initialize visualization components
            self.system_logger.info("Initializing visualization components...")
            self.hud = HUDOverlay()
            self.radar = RadarMapper()
            
            if self.config.show_point_cloud:
                self.system_logger.info("Initializing 3D point cloud visualizer...")
                self.point_cloud = PointCloudVisualizer()
            
            # Initialize audio
            if self.config.audio_enabled:
                self.system_logger.info("Initializing TTS engine...")
                self.tts = TTSEngine()
            
            self.system_logger.info("System initialization complete")
            
        except Exception as e:
            self.logger.log_error_with_context(
                e, 
                {
                    'component': 'system_initialization',
                    'config': str(self.config.__dict__)
                }
            )
            raise RuntimeError(f"Failed to initialize system: {e}")
    
    @log_method_calls(get_logger(), 'StereoVisionSystem')
    def start_system(self):
        """Start the stereo vision system with comprehensive error handling"""
        if self.running:
            self.system_logger.warning("System is already running")
            return
        
        try:
            self.system_logger.info("Starting stereo vision system...")
            
            # Validate system state
            self._validate_system_state()
            
            # Start camera streams
            self.system_logger.info("Starting camera streams...")
            if not self.camera_manager.start_streams():
                raise RuntimeError("Failed to start camera streams")
            
            # Wait for camera initialization
            time.sleep(2.0)
            
            # Verify camera connectivity
            test_frames = self.camera_manager.get_latest_frames()
            if test_frames is None:
                raise RuntimeError("Camera streams not providing frames")
            
            self.system_logger.info("Camera streams validated successfully")
            
            # Start point cloud visualizer if enabled
            if self.point_cloud:
                self.system_logger.info("Starting 3D point cloud visualizer...")
                self.point_cloud.start()
            
            # Start main processing loop
            self.running = True
            self.system_start_time = time.time()
            self.processing_thread = Thread(
                target=self._main_processing_loop, 
                daemon=True,
                name="StereoVisionProcessing"
            )
            self.processing_thread.start()
            
            # Start heartbeat monitoring
            self.heartbeat_thread = Thread(
                target=self._heartbeat_monitor,
                daemon=True,
                name="HeartbeatMonitor"
            )
            self.heartbeat_thread.start()
            
            self.system_logger.info("System started successfully")
            log_info("Stereo Vision System is now running", "StereoVisionSystem")
            
        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    'component': 'system_startup',
                    'running_state': self.running
                }
            )
            self.stop_system()
            raise RuntimeError(f"Failed to start system: {e}")
    
    @log_method_calls(get_logger(), 'StereoVisionSystem')
    def stop_system(self):
        """Stop the stereo vision system gracefully"""
        if not self.running:
            return
        
        self.system_logger.info("Stopping stereo vision system...")
        
        # Signal shutdown
        self.running = False
        self.shutdown_event.set()
        
        # Stop components in reverse order
        try:
            if self.point_cloud:
                self.system_logger.info("Stopping point cloud visualizer...")
                self.point_cloud.stop()
            
            if self.camera_manager:
                self.system_logger.info("Stopping camera streams...")
                self.camera_manager.stop_streams()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.system_logger.info("Waiting for processing thread to finish...")
                self.processing_thread.join(timeout=5.0)
                if self.processing_thread.is_alive():
                    self.system_logger.warning("Processing thread did not stop gracefully")
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Log final statistics
            uptime = time.time() - self.system_start_time
            self.system_logger.info(f"System uptime: {uptime:.1f} seconds")
            self.system_logger.info(f"Total frames processed: {self.total_frames_processed}")
            self.system_logger.info(f"Total detections: {self.total_detections}")
            
            # Force log performance stats
            self.logger.log_performance_stats(force=True)
            
        except Exception as e:
            self.logger.log_error_with_context(e, {'component': 'system_shutdown'})
        
        self.system_logger.info("System stopped")
    
    def _validate_system_state(self):
        """Validate system state before starting"""
        required_components = [
            ('camera_manager', self.camera_manager),
            ('stereo_processor', self.stereo_processor),
            ('detector', self.detector),
            ('tracker', self.tracker),
            ('hud', self.hud),
            ('radar', self.radar)
        ]
        
        for name, component in required_components:
            if component is None:
                raise RuntimeError(f"Required component '{name}' not initialized")
    
    def _heartbeat_monitor(self):
        """Monitor system health and log heartbeat"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat > 30:  # 30 second heartbeat
                    stats = self.perf_monitor.get_current_stats()
                    self.system_logger.info(f"System heartbeat - FPS: {stats.get('fps', 0):.1f}")
                    self.logger.log_performance_stats()
                    self.last_heartbeat = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.log_error_with_context(e, {'component': 'heartbeat_monitor'})
                time.sleep(10)
    
    @log_performance(get_logger(), 'StereoVisionSystem')
    def _main_processing_loop(self):
        """Main processing loop with comprehensive error handling and logging"""
        self.system_logger.info("Starting main processing loop")
        last_alert_time = 0
        frame_count = 0
        
        while self.running and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Check if we should process this frame
                if not self.frame_skipper.should_process_frame():
                    time.sleep(0.001)  # Minimal sleep
                    continue
                
                # Get synchronized frames
                frames = self.camera_manager.get_latest_frames()
                if frames is None:
                    self._handle_frame_error("No frames available from cameras")
                    continue
                
                left_frame, right_frame = frames
                frame_count += 1
                
                # Validate frames
                if not self._validate_frames(left_frame, right_frame):
                    continue
                
                # Update performance monitor
                self.perf_monitor.update_fps()
                self.perf_monitor.update_system_stats()
                
                # Process stereo vision
                disparity = self.stereo_processor.compute_disparity(left_frame, right_frame)
                
                # Detect objects
                detections = self.detector.detect_objects(left_frame)
                
                # Calculate distances for detections
                enhanced_detections = self._enhance_detections_with_distance(detections, disparity)
                
                # Update tracker
                tracked_objects = self.tracker.update(enhanced_detections)
                
                # Log detections if any
                if enhanced_detections:
                    distances = [d.get('distance', 0) for d in enhanced_detections]
                    self.logger.log_detection_event(enhanced_detections, distances)
                    self.total_detections += len(enhanced_detections)
                
                # Create display frame with HUD
                display_frame = self.hud.render_hud(
                    left_frame.copy(), 
                    tracked_objects,
                    self.perf_monitor.get_current_stats()
                )
                
                # Update radar map
                radar_frame = self.radar.update_radar(tracked_objects)
                
                # Update point cloud if enabled
                if self.point_cloud:
                    self.point_cloud.update_point_cloud(
                        disparity, left_frame, 
                        self.stereo_processor.Q_matrix, 
                        enhanced_detections
                    )
                
                # Audio alerts with intelligent timing
                if self.tts and time.time() - last_alert_time > 2.0:
                    alert_message = self._check_alert_conditions(tracked_objects)
                    if alert_message:
                        self.tts.speak(alert_message)
                        last_alert_time = time.time()
                        self.system_logger.info(f"Audio alert: {alert_message}")
                
                # Display frames
                self._display_frames(display_frame, radar_frame, disparity)
                
                # Handle GUI updates
                if self.gui_mode and self.gui_window:
                    stats = self.perf_monitor.get_current_stats()
                    self.gui_window.update_stats(stats)
                
                # Update frame timing
                processing_time = time.time() - start_time
                self.frame_skipper.update_timing(processing_time)
                
                # Handle keyboard input
                if not self._handle_keyboard_input(left_frame, right_frame, tracked_objects):
                    break  # User requested quit
                
                # Update counters
                self.total_frames_processed += 1
                self.consecutive_errors = 0  # Reset error counter on success
                
                # Periodic logging
                if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    self.system_logger.debug(f"Processed {frame_count} frames, {len(enhanced_detections)} detections")
                
            except KeyboardInterrupt:
                self.system_logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self._handle_processing_error(e)
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.system_logger.critical("Too many consecutive errors, stopping system")
                    break
        
        self.system_logger.info("Processing loop ended")
    
    def _validate_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> bool:
        """Validate frame integrity"""
        if left_frame is None or right_frame is None:
            self._handle_frame_error("One or both frames are None")
            return False
        
        if left_frame.size == 0 or right_frame.size == 0:
            self._handle_frame_error("One or both frames are empty")
            return False
        
        if left_frame.shape != right_frame.shape:
            self._handle_frame_error(f"Frame shape mismatch: {left_frame.shape} vs {right_frame.shape}")
            return False
        
        return True
    
    def _enhance_detections_with_distance(self, detections: list, disparity: np.ndarray) -> list:
        """Calculate distances for detections with error handling"""
        enhanced_detections = []
        
        for detection in detections:
            try:
                distance = self.stereo_processor.calculate_distance(
                    detection['center'], disparity
                )
                
                if distance > 0 and distance < self.config.max_detection_distance:
                    detection['distance'] = distance
                    enhanced_detections.append(detection)
                
            except Exception as e:
                log_debug(f"Failed to calculate distance for detection: {e}", "StereoVisionSystem")
        
        return enhanced_detections
    
    def _check_alert_conditions(self, tracked_objects: list) -> Optional[str]:
        """Check if any alert conditions are met"""
        for obj in tracked_objects:
            distance = obj.get('distance', float('inf'))
            label = obj.get('label', '').lower()
            
            if distance < self.config.alert_distance:
                if 'human' in label or 'person' in label:
                    return f"Human detected at {distance:.1f} meters"
                elif 'animal' in label or 'dog' in label or 'cat' in label:
                    return f"Animal detected at {distance:.1f} meters"
        
        return None
    
    def _display_frames(self, display_frame: np.ndarray, radar_frame: Optional[np.ndarray], 
                       disparity: np.ndarray):
        """Display frames with error handling"""
        try:
            if self.config.show_main_view and display_frame is not None:
                cv2.imshow('Stereo Vision System', display_frame)
            
            if self.config.show_radar and radar_frame is not None:
                cv2.imshow('Radar Map', radar_frame)
            
            if self.config.show_disparity and disparity is not None:
                disparity_display = cv2.applyColorMap(
                    cv2.convertScaleAbs(disparity, alpha=0.03), cv2.COLORMAP_JET
                )
                cv2.imshow('Disparity Map', disparity_display)
                
        except Exception as e:
            log_debug(f"Error displaying frames: {e}", "StereoVisionSystem")
    
    def _handle_keyboard_input(self, left_frame: np.ndarray, right_frame: np.ndarray, 
                              tracked_objects: list) -> bool:
        """Handle keyboard input, return False to quit"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.system_logger.info("Quit command received")
            return False
        elif key == ord('s'):
            self.save_snapshot(left_frame, right_frame, tracked_objects)
        elif key == ord('r'):
            self.radar.reset_trails()
            self.system_logger.info("Radar trails reset")
        elif key == ord('p'):
            # Toggle pause (future feature)
            self.system_logger.info("Pause toggle requested")
        elif key == ord('d'):
            # Toggle debug mode
            self.system_logger.info("Debug mode toggle requested")
        
        return True
    
    def _handle_frame_error(self, error_msg: str):
        """Handle frame-related errors"""
        self.consecutive_errors += 1
        log_warning(f"Frame error: {error_msg}", "StereoVisionSystem")
        time.sleep(0.01)
    
    def _handle_processing_error(self, error: Exception):
        """Handle processing errors with context"""
        self.consecutive_errors += 1
        self.logger.log_error_with_context(
            error,
            {
                'component': 'main_processing_loop',
                'consecutive_errors': self.consecutive_errors,
                'total_frames_processed': self.total_frames_processed
            }
        )
        time.sleep(0.1)  # Longer sleep on error
    
    @log_method_calls(get_logger(), 'StereoVisionSystem')
    def save_snapshot(self, left_frame: np.ndarray, right_frame: np.ndarray, 
                     detections: list):
        """Save system snapshot with comprehensive data"""
        try:
            timestamp = int(time.time())
            snapshot_dir = Path('snapshots')
            snapshot_dir.mkdir(exist_ok=True)
            
            # Save frames
            cv2.imwrite(str(snapshot_dir / f'left_{timestamp}.jpg'), left_frame)
            cv2.imwrite(str(snapshot_dir / f'right_{timestamp}.jpg'), right_frame)
            
            # Save detection and system data
            snapshot_data = {
                'timestamp': timestamp,
                'detections': detections,
                'system_stats': self.perf_monitor.get_current_stats(),
                'config': {
                    'baseline_distance': self.config.baseline_distance,
                    'focal_length': self.config.focal_length,
                    'confidence_threshold': self.config.confidence_threshold
                },
                'totals': {
                    'frames_processed': self.total_frames_processed,
                    'total_detections': self.total_detections,
                    'uptime': time.time() - self.system_start_time
                }
            }
            
            with open(snapshot_dir / f'data_{timestamp}.json', 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            self.system_logger.info(f"Snapshot saved: {timestamp}")
            
        except Exception as e:
            self.logger.log_error_with_context(e, {'component': 'save_snapshot'})
    
    @log_method_calls(get_logger(), 'StereoVisionSystem')
    def update_settings(self, settings: Dict[str, Any]):
        """Update system settings from GUI with validation"""
        try:
            self.system_logger.info("Updating system settings...")
            
            # Update camera settings
            if 'camera' in settings:
                camera_settings = settings['camera']
                if 'baseline' in camera_settings:
                    new_baseline = float(camera_settings['baseline'])
                    if 1.0 <= new_baseline <= 100.0:  # Reasonable range
                        self.config.baseline_distance = new_baseline
                        if self.stereo_processor:
                            self.stereo_processor.update_baseline(new_baseline)
                    else:
                        raise ValueError(f"Baseline {new_baseline} out of valid range (1-100 cm)")
            
            # Update detection settings
            if 'detection' in settings:
                det_settings = settings['detection']
                if 'confidence' in det_settings:
                    new_confidence = float(det_settings['confidence'])
                    if 0.1 <= new_confidence <= 1.0:
                        self.config.confidence_threshold = new_confidence
                        if self.detector:
                            self.detector.update_confidence_threshold(new_confidence)
                    else:
                        raise ValueError(f"Confidence {new_confidence} out of valid range (0.1-1.0)")
            
            # Update visualization settings
            if 'visualization' in settings:
                viz_settings = settings['visualization']
                self.config.show_hud = viz_settings.get('show_hud', self.config.show_hud)
                self.config.show_radar = viz_settings.get('show_radar', self.config.show_radar)
                self.config.show_point_cloud = viz_settings.get('show_point_cloud', self.config.show_point_cloud)
            
            self.system_logger.info("Settings updated successfully")
            
        except Exception as e:
            self.logger.log_error_with_context(e, {'component': 'update_settings', 'settings': settings})
            raise

def setup_signal_handlers(system: StereoVisionSystem):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if system:
            system.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point with comprehensive error handling"""
    parser = argparse.ArgumentParser(
        description='Production Stereo Vision System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config/default_config.json',
                       help='Configuration file path')
    parser.add_argument('--gui', action='store_true',
                       help='Launch with GUI control panel')
    parser.add_argument('--baseline', type=float, default=10.0,
                       help='Camera baseline distance in cm')
    parser.add_argument('--left-url', type=str, default='http://192.168.1.100:81/stream',
                       help='Left camera stream URL')
    parser.add_argument('--right-url', type=str, default='http://192.168.1.101:81/stream',
                       help='Right camera stream URL')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    # Initialize logging system
    try:
        logger = init_logger(
            log_dir=args.log_dir,
            log_level=args.log_level,
            console_output=True
        )
        log_info(f"Stereo Vision System starting...", "Main")
        log_info(f"Command line args: {vars(args)}", "Main")
        
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        sys.exit(1)
    
    system = None
    
    try:
        # Load configuration
        config = SystemConfig()
        config.baseline_distance = args.baseline
        config.left_camera_url = args.left_url  
        config.right_camera_url = args.right_url
        
        if args.gui:
            # Launch with GUI
            try:
                from PyQt6.QtWidgets import QApplication
                from gui.control_panel import ControlPanelGUI
                
                app = QApplication(sys.argv)
                gui_window = ControlPanelGUI()
                
                # Create system
                system = StereoVisionSystem(config, gui_mode=True)
                system.gui_app = app
                system.gui_window = gui_window
                
                # Connect GUI signals
                gui_window.settings_changed.connect(system.update_settings)
                gui_window.start_system.connect(system.start_system)
                gui_window.stop_system.connect(system.stop_system)
                
                # Setup signal handlers
                setup_signal_handlers(system)
                
                # Show GUI
                gui_window.show()
                log_info("GUI launched successfully", "Main")
                
                # Run application
                sys.exit(app.exec())
                
            except ImportError as e:
                log_error(f"Failed to import GUI components: {e}", "Main")
                print("GUI mode requires PyQt6. Install with: pip install PyQt6")
                sys.exit(1)
            
        else:
            # Command line mode
            system = StereoVisionSystem(config)
            
            # Setup signal handlers
            setup_signal_handlers(system)
            
            print("\n" + "="*60)
            print("STEREO VISION SYSTEM - PRODUCTION MODE")
            print("="*60)
            print("Controls:")
            print("  'q' - Quit system")
            print("  's' - Save snapshot")
            print("  'r' - Reset radar trails")
            print("  'p' - Toggle pause (future)")
            print("  'd' - Toggle debug mode (future)")
            print(f"\nConfiguration:")
            print(f"  Left Camera:  {args.left_url}")
            print(f"  Right Camera: {args.right_url}")
            print(f"  Baseline:     {args.baseline} cm")
            print(f"  Log Level:    {args.log_level}")
            print(f"  Log Directory: {args.log_dir}")
            print("="*60)
            
            # Start system
            system.start_system()
            
            try:
                # Keep main thread alive
                while system.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received...")
            finally:
                system.stop_system()
                cv2.destroyAllWindows()
                log_info("System shutdown complete", "Main")
    
    except Exception as e:
        logger_instance = get_logger()
        logger_instance.log_error_with_context(e, {'component': 'main_entry_point'})
        print(f"Critical system error: {e}")
        if system:
            system.stop_system()
        sys.exit(1)

if __name__ == "__main__":
    main()