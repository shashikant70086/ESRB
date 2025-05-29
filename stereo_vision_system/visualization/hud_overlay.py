# visualization/hud_overlay.py
import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Optional
from core.tracker_system import TrackedObject

class HUDOverlay:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.start_time = time.time()
        
        # Colors (BGR format)
        self.colors = {
            'human': (0, 255, 0),      # Green
            'animal': (0, 255, 255),   # Yellow
            'vehicle': (0, 165, 255),  # Orange
            'object': (255, 0, 0),     # Blue
            'unknown': (128, 128, 128), # Gray
            'danger': (0, 0, 255),     # Red
            'warning': (0, 255, 255),  # Yellow
            'safe': (0, 255, 0),       # Green
            'scan_line': (0, 255, 255), # Cyan
            'crosshair': (255, 255, 255), # White
            'text': (255, 255, 255),   # White
            'background': (0, 0, 0)    # Black
        }
        
        # Animation parameters
        self.scan_angle = 0
        self.pulse_intensity = 0
        self.grid_alpha = 0.3
        
    def _get_threat_level_color(self, distance: float, class_name: str) -> Tuple[int, int, int]:
        """Determine color based on distance and object type"""
        if class_name.lower() in ['human', 'person']:
            if distance < 2.0:
                return self.colors['danger']
            elif distance < 5.0:
                return self.colors['warning']
            else:
                return self.colors['safe']
        elif class_name.lower() in ['animal', 'dog', 'cat', 'bird']:
            return self.colors['animal']
        elif class_name.lower() in ['car', 'truck', 'motorcycle', 'vehicle']:
            return self.colors['vehicle']
        else:
            return self.colors['object']
    
    def _draw_animated_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                          color: Tuple[int, int, int], thickness: int = 2):
        """Draw animated bounding box with corner brackets"""
        x, y, w, h = bbox
        corner_length = min(w, h) // 6
        
        # Animate corner brackets
        pulse = int(20 * (1 + math.sin(self.frame_count * 0.2)))
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness + pulse//10)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness + pulse//10)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness + pulse//10)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness + pulse//10)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness + pulse//10)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness + pulse//10)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness + pulse//10)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness + pulse//10)
        
        # Center crosshair
        center_x, center_y = x + w//2, y + h//2
        cross_size = 10
        cv2.line(frame, (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), color, 1)
        cv2.line(frame, (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), color, 1)
    
    def _draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                        color: Tuple[int, int, int]):
        """Draw object trajectory path"""
        if len(trajectory) < 2:
            return
        
        # Draw trajectory lines with fading effect
        for i in range(1, len(trajectory)):
            alpha = i / len(trajectory)
            thickness = max(1, int(3 * alpha))
            
            # Create fading effect by blending with background
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, trajectory[i-1], trajectory[i], fade_color, thickness)
    
    def _draw_distance_indicator(self, frame: np.ndarray, center: Tuple[int, int], 
                               distance: float, color: Tuple[int, int, int]):
        """Draw distance measurement with visual indicator"""
        x, y = center
        
        # Distance text
        distance_text = f"{distance:.1f}m"
        text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        padding = 5
        rect_x = x - text_size[0]//2 - padding
        rect_y = y - 40 - text_size[1] - padding
        rect_w = text_size[0] + 2*padding
        rect_h = text_size[1] + 2*padding
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Distance text
        cv2.putText(frame, distance_text, (rect_x + padding, rect_y + text_size[1] + padding), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Distance line indicator
        line_length = min(50, int(distance * 10))
        cv2.line(frame, (x, y - 20), (x, y - 20 - line_length), color, 2)
    
    def _draw_scanning_effect(self, frame: np.ndarray):
        """Draw animated scanning lines"""
        # Horizontal scanning line
        scan_y = int(self.height * (0.5 + 0.4 * math.sin(self.frame_count * 0.1)))
        cv2.line(frame, (0, scan_y), (self.width, scan_y), self.colors['scan_line'], 1)
        
        # Vertical scanning line
        scan_x = int(self.width * (0.5 + 0.4 * math.cos(self.frame_count * 0.15)))
        cv2.line(frame, (scan_x, 0), (scan_x, self.height), self.colors['scan_line'], 1)
        
        # Radial sweep
        center = (self.width//2, self.height//2)
        radius = min(self.width, self.height) // 4
        angle = (self.frame_count * 5) % 360
        end_x = int(center[0] + radius * math.cos(math.radians(angle)))
        end_y = int(center[1] + radius * math.sin(math.radians(angle)))
        cv2.line(frame, center, (end_x, end_y), self.colors['scan_line'], 2)
    
    def _draw_hud_elements(self, frame: np.ndarray, objects: List[TrackedObject]):
        """Draw main HUD elements"""
        # Frame rate and system info
        fps = self.frame_count / (time.time() - self.start_time + 0.001)
        status_text = f"FPS: {fps:.1f} | Objects: {len(objects)} | STEREO VISION ACTIVE"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.colors['text'], 2)
        
        # Crosshair at center
        center = (self.width//2, self.height//2)
        cv2.line(frame, (center[0] - 20, center[1]), (center[0] + 20, center[1]), 
                self.colors['crosshair'], 2)
        cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), 
                self.colors['crosshair'], 2)
        cv2.circle(frame, center, 5, self.colors['crosshair'], 2)
        
        # Distance grid overlay
        self._draw_distance_grid(frame)
    
    def _draw_distance_grid(self, frame: np.ndarray):
        """Draw distance measurement grid"""
        overlay = frame.copy()
        
        # Horizontal grid lines (every 2 meters assuming calibration)
        for i in range(1, 6):  # Up to 10 meters
            y = self.height - (i * self.height // 10)
            cv2.line(overlay, (0, y), (self.width, y), self.colors['scan_line'], 1)
            cv2.putText(overlay, f"{i*2}m", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.colors['text'], 1)
        
        # Vertical reference lines
        for i in range(1, 5):
            x = i * self.width // 4
            cv2.line(overlay, (x, 0), (x, self.height), self.colors['scan_line'], 1)
        
        # Blend grid with main frame
        cv2.addWeighted(overlay, self.grid_alpha, frame, 1 - self.grid_alpha, 0, frame)
    
    def _draw_object_info_panel(self, frame: np.ndarray, obj: TrackedObject, 
                              panel_x: int, panel_y: int):
        """Draw detailed object information panel"""
        info_lines = [
            f"ID: {obj.id:03d}",
            f"Type: {obj.class_name.upper()}",
            f"Distance: {obj.distance:.2f}m",
            f"Confidence: {obj.confidence:.0%}",
            f"Trajectory: {len(obj.trajectory)} pts"
        ]
        
        # Panel background
        panel_width = 200
        panel_height = len(info_lines) * 25 + 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        color = self._get_threat_level_color(obj.distance, obj.class_name)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), color, 2)
        
        # Info text
        for i, line in enumerate(info_lines):
            y_pos = panel_y + 20 + i * 25
            cv2.putText(frame, line, (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def render(self, frame: np.ndarray, objects: List[TrackedObject]) -> np.ndarray:
        """
        Render complete HUD overlay on frame
        
        Args:
            frame: Input frame
            objects: List of tracked objects
            
        Returns:
            Frame with HUD overlay
        """
        self.frame_count += 1
        hud_frame = frame.copy()
        
        # Draw scanning effects
        self._draw_scanning_effect(hud_frame)
        
        # Draw HUD elements
        self._draw_hud_elements(hud_frame, objects)
        
        # Draw objects
        info_panel_y = 60
        for i, obj in enumerate(objects):
            color = self._get_threat_level_color(obj.distance, obj.class_name)
            
            # Animated bounding box
            self._draw_animated_box(hud_frame, obj.bbox, color)
            
            # Trajectory
            self._draw_trajectory(hud_frame, obj.trajectory, color)
            
            # Distance indicator
            self._draw_distance_indicator(hud_frame, obj.center, obj.distance, color)
            
            # Object label
            label = f"{obj.class_name.upper()} #{obj.id:03d}"
            label_pos = (obj.bbox[0], obj.bbox[1] - 10)
            cv2.putText(hud_frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Detailed info panel for first few objects
            if i < 3:  # Show detailed info for first 3 objects
                self._draw_object_info_panel(hud_frame, obj, self.width - 220, info_panel_y)
                info_panel_y += 150
        
        return hud_frame
    
    def set_grid_alpha(self, alpha: float):
        """Set grid transparency (0.0 to 1.0)"""
        self.grid_alpha = max(0.0, min(1.0, alpha))
    
    def reset_frame_count(self):
        """Reset frame counter and start time"""
        self.frame_count = 0
        self.start_time = time.time()