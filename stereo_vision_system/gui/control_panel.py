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
        
        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self.calibrate_cameras.emit)
        button_layout.addWidget(self.calibrate_button)
        
        self.load_calib_button = QPushButton("Load Calibration")
        self.load_calib_button.clicked.connect(self.load_calibration)
        button_layout.addWidget(self.load_calib_button)
        
        self.save_calib_button = QPushButton("Save Calibration")
        self.save_calib_button.clicked.connect(self.save_calibration)
        button_layout.addWidget(self.save_calib_button)
        
        calib_layout.addLayout(button_layout)
        
        # Calibration status
        self.calib_status = QLabel("No calibration loaded")
        self.calib_status.setStyleSheet("color: orange;")
        calib_layout.addWidget(self.calib_status)
        
        layout.addWidget(calib_group)
        
        # Manual parameters
        manual_group = QGroupBox("Manual Parameters")
        manual_layout = QGridLayout(manual_group)
        
        manual_layout.addWidget(QLabel("Focal Length (px):"), 0, 0)
        self.focal_length = QDoubleSpinBox()
        self.focal_length.setRange(100.0, 2000.0)
        self.focal_length.setValue(500.0)
        manual_layout.addWidget(self.focal_length, 0, 1)
        
        manual_layout.addWidget(QLabel("Principal Point X:"), 1, 0)
        self.cx = QDoubleSpinBox()
        self.cx.setRange(0.0, 1000.0)
        self.cx.setValue(320.0)
        manual_layout.addWidget(self.cx, 1, 1)
        
        manual_layout.addWidget(QLabel("Principal Point Y:"), 2, 0)
        self.cy = QDoubleSpinBox()
        self.cy.setRange(0.0, 1000.0)
        self.cy.setValue(240.0)
        manual_layout.addWidget(self.cy, 2, 1)
        
        layout.addWidget(manual_group)
        layout.addStretch()
        
        self.tabs.addTab(calib_widget, "Calibration")
    
    def setup_timer(self):
        """Setup timer for periodic updates"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update every 100ms
    
    def toggle_system(self):
        """Toggle system start/stop"""
        if not self.system_running:
            # Start system
            settings = self.get_current_settings()
            self.settings_changed.emit(settings)
            self.start_system.emit()
            self.start_button.setText("Stop System")
            self.system_running = True
            self.statusBar().showMessage("System Running")
        else:
            # Stop system
            self.stop_system.emit()
            self.start_button.setText("Start System")
            self.system_running = False
            self.statusBar().showMessage("System Stopped")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings from UI"""
        return {
            'camera': {
                'left_url': self.left_url_input.toPlainText().strip(),
                'right_url': self.right_url_input.toPlainText().strip(),
                'baseline': self.baseline_input.value(),
                'left_rotation': self.left_rotation.currentText(),
                'right_rotation': self.right_rotation.currentText(),
            },
            'detection': {
                'model': self.model_combo.currentText(),
                'confidence': self.confidence_slider.value() / 100.0,
                'max_distance': self.max_distance.value(),
            },
            'audio': {
                'enabled': self.audio_enabled.isChecked(),
                'alert_distance': self.alert_distance.value(),
            },
            'visualization': {
                'show_hud': self.show_hud.isChecked(),
                'show_distances': self.show_distances.isChecked(),
                'show_fps': self.show_fps.isChecked(),
                'show_scan_lines': self.show_scan_lines.isChecked(),
                'show_radar': self.show_radar.isChecked(),
                'show_point_cloud': self.show_point_cloud.isChecked(),
                'show_disparity': self.show_disparity.isChecked(),
            },
            'performance': {
                'target_fps': self.target_fps.value(),
                'use_gpu': self.use_gpu.isChecked(),
                'adaptive_fps': self.adaptive_fps.isChecked(),
            },
            'calibration': {
                'focal_length': self.focal_length.value(),
                'cx': self.cx.value(),
                'cy': self.cy.value(),
            }
        }
    
    @pyqtSlot(dict)
    def update_stats(self, stats: Dict[str, Any]):
        """Update performance statistics"""
        self.current_stats = stats
    
    def update_display(self):
        """Update display elements"""
        if self.current_stats:
            # Update FPS
            fps = self.current_stats.get('fps', 0.0)
            self.fps_label.setText(f"{fps:.1f}")
            
            # Update progress bars
            cpu = self.current_stats.get('cpu_percent', 0)
            memory = self.current_stats.get('memory_percent', 0)
            gpu = self.current_stats.get('gpu_percent', 0)
            
            self.cpu_progress.setValue(int(cpu))
            self.memory_progress.setValue(int(memory))
            self.gpu_progress.setValue(int(gpu))
            
            # Color code FPS based on performance
            if fps >= 15:
                self.fps_label.setStyleSheet("font-weight: bold; color: green;")
            elif fps >= 10:
                self.fps_label.setStyleSheet("font-weight: bold; color: orange;")
            else:
                self.fps_label.setStyleSheet("font-weight: bold; color: red;")
    
    def load_calibration(self):
        """Load calibration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    calib_data = json.load(f)
                
                # Update UI with loaded values
                if 'focal_length' in calib_data:
                    self.focal_length.setValue(calib_data['focal_length'])
                if 'cx' in calib_data:
                    self.cx.setValue(calib_data['cx'])
                if 'cy' in calib_data:
                    self.cy.setValue(calib_data['cy'])
                
                self.calib_status.setText("Calibration loaded successfully")
                self.calib_status.setStyleSheet("color: green;")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load calibration: {e}")
    
    def save_calibration(self):
        """Save calibration file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "calibration.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import json
                calib_data = {
                    'focal_length': self.focal_length.value(),
                    'cx': self.cx.value(),
                    'cy': self.cy.value(),
                    'baseline': self.baseline_input.value(),
                }
                
                with open(file_path, 'w') as f:
                    json.dump(calib_data, f, indent=2)
                
                self.calib_status.setText("Calibration saved successfully")
                self.calib_status.setStyleSheet("color: green;")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save calibration: {e}")

def run_gui():
    """Run the GUI application"""
    app = QApplication(sys.argv)
    window = ControlPanelGUI()
    window.show()
    return app, window

if __name__ == "__main__":
    app, window = run_gui()
    sys.exit(app.exec())