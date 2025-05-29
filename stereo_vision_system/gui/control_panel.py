# gui/control_panel.py
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QSlider, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QCheckBox, QTextEdit, QGroupBox, QProgressBar,
                             QTabWidget, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
from typing import Dict, Any, Callable

class ControlPanelGUI(QMainWindow):
    """Main control interface for stereo vision system"""
    
    # Signals for communication with main system
    settings_changed = pyqtSignal(dict)
    start_system = pyqtSignal()
    stop_system = pyqtSignal()
    save_snapshot = pyqtSignal()
    calibrate_cameras = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.system_running = False
        self.current_stats = {}
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Stereo Vision Control Panel")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.create_camera_tab()
        self.create_detection_tab()
        self.create_visualization_tab()
        self.create_performance_tab()
        self.create_calibration_tab()
        
        # Status bar
        self.statusBar().showMessage("System Ready")
        
    def create_camera_tab(self):
        """Create camera configuration tab"""
        camera_widget = QWidget()
        layout = QVBoxLayout(camera_widget)
        
        # Camera URLs
        url_group = QGroupBox("Camera Configuration")
        url_layout = QGridLayout(url_group)
        
        url_layout.addWidget(QLabel("Left Camera URL:"), 0, 0)
        self.left_url_input = QTextEdit()
        self.left_url_input.setMaximumHeight(30)
        self.left_url_input.setText("http://192.168.1.100:81/stream")
        url_layout.addWidget(self.left_url_input, 0, 1)
        
        url_layout.addWidget(QLabel("Right Camera URL:"), 1, 0)
        self.right_url_input = QTextEdit()
        self.right_url_input.setMaximumHeight(30)
        self.right_url_input.setText("http://192.168.1.101:81/stream")
        url_layout.addWidget(self.right_url_input, 1, 1)
        
        layout.addWidget(url_group)
        
        # Baseline distance
        baseline_group = QGroupBox("Stereo Configuration")
        baseline_layout = QGridLayout(baseline_group)
        
        baseline_layout.addWidget(QLabel("Baseline Distance (cm):"), 0, 0)
        self.baseline_input = QDoubleSpinBox()
        self.baseline_input.setRange(1.0, 100.0)
        self.baseline_input.setValue(10.0)
        self.baseline_input.setSuffix(" cm")
        baseline_layout.addWidget(self.baseline_input, 0, 1)
        
        # Camera rotations
        baseline_layout.addWidget(QLabel("Left Camera Rotation:"), 1, 0)
        self.left_rotation = QComboBox()
        self.left_rotation.addItems(["0°", "90°", "180°", "270°", "Mirror"])
        baseline_layout.addWidget(self.left_rotation, 1, 1)
        
        baseline_layout.addWidget(QLabel("Right Camera Rotation:"), 2, 0)
        self.right_rotation = QComboBox()
        self.right_rotation.addItems(["0°", "90°", "180°", "270°", "Mirror"])
        baseline_layout.addWidget(self.right_rotation, 2, 1)
        
        layout.addWidget(baseline_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start System")
        self.start_button.clicked.connect(self.toggle_system)
        button_layout.addWidget(self.start_button)
        
        self.snapshot_button = QPushButton("Save Snapshot")
        self.snapshot_button.clicked.connect(self.save_snapshot.emit)
        button_layout.addWidget(self.snapshot_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.tabs.addTab(camera_widget, "Cameras")
        
    def create_detection_tab(self):
        """Create detection configuration tab"""
        detection_widget = QWidget()
        layout = QVBoxLayout(detection_widget)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout(detection_group)
        
        # Model selection
        detection_layout.addWidget(QLabel("Detection Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MediaPipe", "YOLOv8", "Both"])
        detection_layout.addWidget(self.model_combo, 0, 1)
        
        # Confidence threshold
        detection_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        detection_layout.addWidget(self.confidence_slider, 1, 1)
        detection_layout.addWidget(self.confidence_label, 1, 2)
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )
        
        # Distance filter
        detection_layout.addWidget(QLabel("Max Distance (m):"), 2, 0)
        self.max_distance = QDoubleSpinBox()
        self.max_distance.setRange(1.0, 50.0)
        self.max_distance.setValue(10.0)
        self.max_distance.setSuffix(" m")
        detection_layout.addWidget(self.max_distance, 2, 1)
        
        layout.addWidget(detection_group)
        
        # Audio settings
        audio_group = QGroupBox("Audio Feedback")
        audio_layout = QGridLayout(audio_group)
        
        self.audio_enabled = QCheckBox("Enable Audio Alerts")
        self.audio_enabled.setChecked(True)
        audio_layout.addWidget(self.audio_enabled, 0, 0)
        
        audio_layout.addWidget(QLabel("Alert Distance (m):"), 1, 0)
        self.alert_distance = QDoubleSpinBox()
        self.alert_distance.setRange(0.5, 10.0)
        self.alert_distance.setValue(3.0)
        self.alert_distance.setSuffix(" m")
        audio_layout.addWidget(self.alert_distance, 1, 1)
        
        layout.addWidget(audio_group)
        layout.addStretch()
        
        self.tabs.addTab(detection_widget, "Detection")
        
    def create_visualization_tab(self):
        """Create visualization settings tab"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # HUD settings
        hud_group = QGroupBox("HUD Display")
        hud_layout = QGridLayout(hud_group)
        
        self.show_hud = QCheckBox("Show HUD Overlay")
        self.show_hud.setChecked(True)
        hud_layout.addWidget(self.show_hud, 0, 0)
        
        self.show_distances = QCheckBox("Show Distance Labels")
        self.show_distances.setChecked(True)
        hud_layout.addWidget(self.show_distances, 0, 1)
        
        self.show_fps = QCheckBox("Show FPS Counter")
        self.show_fps.setChecked(True)
        hud_layout.addWidget(self.show_fps, 1, 0)
        
        self.show_scan_lines = QCheckBox("Show Scan Lines")
        self.show_scan_lines.setChecked(True)
        hud_layout.addWidget(self.show_scan_lines, 1, 1)
        
        layout.addWidget(hud_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout(display_group)
        
        self.show_radar = QCheckBox("Show 2D Radar")
        self.show_radar.setChecked(True)
        display_layout.addWidget(self.show_radar, 0, 0)
        
        self.show_point_cloud = QCheckBox("Show 3D Point Cloud")
        self.show_point_cloud.setChecked(False)
        display_layout.addWidget(self.show_point_cloud, 0, 1)
        
        self.show_disparity = QCheckBox("Show Disparity Map")
        self.show_disparity.setChecked(False)
        display_layout.addWidget(self.show_disparity, 1, 0)
        
        layout.addWidget(display_group)
        layout.addStretch()
        
        self.tabs.addTab(viz_widget, "Visualization")
        
    def create_performance_tab(self):
        """Create performance monitoring tab"""
        perf_widget = QWidget()
        layout = QVBoxLayout(perf_widget)
        
        # Performance stats
        stats_group = QGroupBox("System Performance")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet("font-weight: bold; color: green;")
        stats_layout.addWidget(self.fps_label, 0, 1)
        
        stats_layout.addWidget(QLabel("CPU Usage:"), 1, 0)
        self.cpu_progress = QProgressBar()
        stats_layout.addWidget(self.cpu_progress, 1, 1)
        
        stats_layout.addWidget(QLabel("Memory Usage:"), 2, 0)
        self.memory_progress = QProgressBar()
        stats_layout.addWidget(self.memory_progress, 2, 1)
        
        stats_layout.addWidget(QLabel("GPU Usage:"), 3, 0)
        self.gpu_progress = QProgressBar()
        stats_layout.addWidget(self.gpu_progress, 3, 1)
        
        layout.addWidget(stats_group)
        
        # Performance settings
        perf_settings_group = QGroupBox("Performance Settings")
        perf_settings_layout = QGridLayout(perf_settings_group)
        
        perf_settings_layout.addWidget(QLabel("Target FPS:"), 0, 0)
        self.target_fps = QSpinBox()
        self.target_fps.setRange(5, 60)
        self.target_fps.setValue(20)
        perf_settings_layout.addWidget(self.target_fps, 0, 1)
        
        self.use_gpu = QCheckBox("Use GPU Acceleration")
        self.use_gpu.setChecked(True)
        perf_settings_layout.addWidget(self.use_gpu, 1, 0)
        
        self.adaptive_fps = QCheckBox("Adaptive Frame Rate")
        self.adaptive_fps.setChecked(True)
        perf_settings_layout.addWidget(self.adaptive_fps, 1, 1)
        
        layout.addWidget(perf_settings_group)
        layout.addStretch()
        
        self.tabs.addTab(perf_widget, "Performance")
        
    def create_calibration_tab(self):
        """Create camera calibration tab"""
        calib_widget = QWidget()
        layout = QVBoxLayout(calib_widget)
        
        # Calibration controls
        calib_group = QGroupBox("Camera Calibration")
        calib_layout = QVBoxLayout(calib_group)
        
        info_label = QLabel(
            "Camera calibration improves depth accuracy.\n"
            "Use a checkerboard pattern for calibration."
        )
        calib_layout.addWidget(info_label)
        
        button_layout = QHBoxLayout()
        
        self.calibrate