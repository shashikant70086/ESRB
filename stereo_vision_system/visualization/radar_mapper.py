# visualization/radar_mapper.py
import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Optional
from core.tracker_system import TrackedObject

class RadarMapper:
    def __init__(self, radar_size: int = 400, max_range: float = 15.0):
        self.radar_size = radar_size
        self.max_range = max_range  # Maximum detection range in meters
        self.center = (radar_size // 2, radar_size // 2)
        self.frame_count = 0
        
        # Colors (BGR format)
        self.colors = {
            'background': (20, 20, 20),
            'grid': (0, 100, 0),
            'sweep': (0, 255, 255),
            'human': (0, 255, 0),
            'animal': (0, 255, 255),
            'vehicle': (0, 165, 255),
            'object': (255, 0, 0),
            'unknown': (128, 128, 128),
            'text': (255, 255, 255),
            'center': (255, 255, 255),
            'trail': (0, 150, 0)
        }
        
        # Radar parameters
        self.sweep_angle = 0
        self.object_trails = {}  # Store object position history
        self.max_trail_length = 20
        
    def _world_to_radar(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert world coordinates to radar screen coordinates
        
        Args:
            world_pos: (x, y) position in meters relative to camera
            
        Returns:
            (x, y) pixel coordinates on radar screen
        """
        world_x, world_y = world_pos
        
        # Scale to radar range
        scale = (self.radar_size // 2) / self.max_range
        
        # Convert to radar coordinates (y-axis flipped for screen coordinates)
        radar_x = int(self.center[0] + world_x * scale)
        radar_y = int(self.center[1] - world_y * scale)  # Flip Y for screen coords
        
        # Clamp to radar bounds
        radar_x = max(0, min(self.radar_size - 1, radar_x))
        radar_y = max(0, min(self.radar_size - 1, radar_y))
        
        return (radar_x, radar_y)
    
    def _draw_radar_grid(self, radar_img: np.ndarray):
        """Draw radar grid and range circles"""
        # Range circles
        for i in range(1, 6):
            radius = int((i * self.radar_size // 2) / 5)
            cv2.circle(radar_img, self.center, radius, self.colors['grid'], 1)
            
            # Range labels
            range_text = f"{i * self.max_range / 5:.0f}m"
            text_pos = (self.center[0] + radius - 20, self.center[1] - 5)
            cv2.putText(radar_img, range_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.colors['text'], 1)
        
        # Cardinal direction lines
        # North-South line
        cv2.line(radar_img, (self.center[0], 0), 
                (self.center[0], self.radar_size), self.colors['grid'], 1)
        # East-West line
        cv2.line(radar_img, (0, self.center[1]), 
                (self.radar_size, self.center[1]), self.colors['grid'], 1)
        
        # Diagonal lines
        cv2.line(radar_img, (0, 0), (self.radar_size, self.radar_size), self.colors['grid'], 1)
        cv2.line(radar_img, (self.radar_size, 0), (0, self.radar_size), self.colors['grid'], 1)
        
        # Direction labels
        cv2.putText(radar_img, "N", (self.center[0] - 8, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        cv2.putText(radar_img, "S", (self.center[0] - 8, self.radar_size - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        cv2.putText(radar_img, "E", (self.radar_size - 15, self.center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        cv2.putText(radar_img, "W", (5, self.center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
    
    def _draw_sweep_line(self, radar_img: np.ndarray):
        """Draw animated radar sweep line"""
        # Update sweep angle
        self.sweep_angle = (self.sweep_angle + 3) % 360
        
        # Calculate sweep line end point
        sweep_radius = self.radar_size // 2
        end_x = int(self.center[0] + sweep_radius * math.cos(math.radians(self.sweep_angle)))
        end_y = int(self.center[1] + sweep_radius * math.sin(math.radians(self.sweep_angle)))
        
        # Draw sweep line with fading effect
        cv2.line(radar_img, self.center, (end_x, end_y), self.colors['sweep'], 2)
        
        # Draw sweep arc for visual effect
        for i in range(10):
            alpha = (10 - i) / 10.0
            arc_color = tuple(int(c * alpha) for c in self.colors['sweep'])
            arc_angle = self.sweep_angle - i * 3
            
            arc_end_x = int(self.center[0] + sweep_radius * math.cos(math.radians(arc_angle)))
            arc_end_y = int(self.center[1] + sweep_radius * math.sin(math.radians(arc_angle)))
            
            cv2.line(radar_img, self.center, (arc_end_x, arc_end_y), arc_color, 1)
    
    def _get_object_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for object based on class name"""
        class_lower = class_name.lower()
        if class_lower in ['human', 'person']:
            return self.colors['human']
        elif class_lower in ['animal', 'dog', 'cat', 'bird']:
            return self.colors['animal']
        elif class_lower in ['car', 'truck', 'motorcycle', 'vehicle']:
            return self.colors['vehicle']
        elif class_lower in ['object']:
            return self.colors['object']
        else:
            return self.colors['unknown']
    
    def _update_object_trail(self, obj_id: int, position: Tuple[int, int]):
        """Update object movement trail"""
        if obj_id not in self.object_trails:
            self.object_trails[obj_id] = []
        
        self.object_trails[obj_id].append(position)
        
        # Limit trail length
        if len(self.object_trails[obj_id]) > self.max_trail_length:
            self.object_trails[obj_id] = self.object_trails[obj_id][-self.max_trail_length:]
    
    def _draw_object_trail(self, radar_img: np.ndarray, obj_id: int, color: Tuple[int, int, int]):
        """Draw object movement trail"""
        if obj_id not in self.object_trails or len(self.object_trails[obj_id]) < 2:
            return
        
        trail = self.object_trails[obj_id]
        
        # Draw trail with fading effect
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            trail_color = tuple(int(c * alpha * 0.7) for c in color)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(radar_img, trail[i-1], trail[i], trail_color, thickness)
    
    def _estimate_world_position(self, obj: TrackedObject, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """
        Estimate world position from object center and distance
        
        This is a simplified estimation. In a real system, you'd use:
        - Proper camera calibration
        - Stereo vision triangulation
        - Known camera field of view
        """
        # Get object center in image coordinates
        center_x, center_y = obj.center
        
        # Convert to normalized coordinates (-1 to 1)
        norm_x = (center_x - frame_width/2) / (frame_width/2)
        norm_y = (center_y - frame_height/2) / (frame_height/2)
        
        # Estimate horizontal angle (assuming 60° FOV)
        horizontal_angle = norm_x * 30  # Half of 60° FOV
        
        # Calculate world coordinates
        world_x = obj.distance * math.sin(math.radians(horizontal_angle))
        world_y = obj.distance * math.cos(math.radians(horizontal_angle))
        
        return (world_x, world_y)
    
    def _draw_object_on_radar(self, radar_img: np.ndarray, obj: TrackedObject, 
                            radar_pos: Tuple[int, int]):
        """Draw individual object on radar"""
        color = self._get_object_color(obj.class_name)
        
        # Draw object trail
        self._draw_object_trail(radar_img, obj.id, color)
        
        # Draw object dot
        cv2.circle(radar_img, radar_pos, 6, color, -1)
        cv2.circle(radar_img, radar_pos, 8, color, 2)
        
        # Draw object ID
        id_text = str(obj.id)
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_pos = (radar_pos[0] - text_size[0]//2, radar_pos[1] + 20)
        
        # Text background
        cv2.rectangle(radar_img, 
                     (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                     (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                     self.colors['background'], -1)
        
        cv2.putText(radar_img, id_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, color, 1)
        
        # Distance indicator (pulsing circle)
        pulse = int(5 * (1 + math.sin(self.frame_count * 0.3 + obj.id)))
        cv2.circle(radar_img, radar_pos, 12 + pulse, color, 1)
    
    def _draw_radar_info(self, radar_img: np.ndarray, objects: List[TrackedObject]):
        """Draw radar information panel"""
        info_y = 20
        
        # Title
        cv2.putText(radar_img, "RADAR VIEW", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        info_y += 25
        
        # Range info
        cv2.putText(radar_img, f"Range: {self.max_range:.0f}m", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        info_y += 20
        
        # Object count
        cv2.putText(radar_img, f"Objects: {len(objects)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        info_y += 25
        
        # Legend
        legend_items = [
            ("Human", self.colors['human']),
            ("Animal", self.colors['animal']),
            ("Vehicle", self.colors['vehicle']),
            ("Object", self.colors['object'])
        ]
        
        cv2.putText(radar_img, "LEGEND:", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        info_y += 15
        
        for label, color in legend_items:
            cv2.circle(radar_img, (15, info_y), 4, color, -1)
            cv2.putText(radar_img, label, (25, info_y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
            info_y += 15
    
    def render(self, objects: List[TrackedObject], frame_width: int, frame_height: int) -> np.ndarray:
        """
        Render radar map
        
        Args:
            objects: List of tracked objects
            frame_width: Width of camera frame (for position estimation)
            frame_height: Height of camera frame (for position estimation)
            
        Returns:
            Radar image
        """
        self.frame_count += 1
        
        # Create radar background
        radar_img = np.full((self.radar_size, self.radar_size, 3), 
                           self.colors['background'], dtype=np.uint8)
        
        # Draw radar grid
        self._draw_radar_grid(radar_img)
        
        # Draw sweep line
        self._draw_sweep_line(radar_img)
        
        # Draw center point
        cv2.circle(radar_img, self.center, 3, self.colors['center'], -1)
        cv2.putText(radar_img, "ORIGIN", (self.center[0] - 25, self.center[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
        
        # Draw objects
        for obj in objects:
            # Estimate world position
            world_pos = self._estimate_world_position(obj, frame_width, frame_height)
            
            # Convert to radar coordinates
            radar_pos = self._world_to_radar(world_pos)
            
            # Update trail
            self._update_object_trail(obj.id, radar_pos)
            
            # Draw object
            self._draw_object_on_radar(radar_img, obj, radar_pos)
        
        # Draw info panel
        self._draw_radar_info(radar_img, objects)
        
        return radar_img
    
    def set_max_range(self, max_range: float):
        """Set maximum radar range in meters"""
        self.max_range = max(1.0, max_range)
    
    def clear_trails(self):
        """Clear all object trails"""
        self.object_trails.clear()
    
    def get_radar_stats(self) -> Dict:
        """Get radar statistics"""
        return {
            'radar_size': self.radar_size,
            'max_range': self.max_range,
            'active_trails': len(self.object_trails),
            'frame_count': self.frame_count
        }