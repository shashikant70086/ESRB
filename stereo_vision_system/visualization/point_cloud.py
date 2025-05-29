# visualization/point_cloud.py
import numpy as np
import open3d as o3d
import cv2
from threading import Thread, Lock
import time
from typing import Optional, Tuple, List

class PointCloudVisualizer:
    """Real-time 3D point cloud visualization using Open3D"""
    
    def __init__(self, max_points: int = 50000):
        self.max_points = max_points
        self.vis = None
        self.pcd = None
        self.running = False
        self.lock = Lock()
        self.current_points = None
        self.current_colors = None
        
    def initialize_visualizer(self):
        """Initialize Open3D visualizer"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="3D Point Cloud", width=800, height=600)
            
            # Create initial empty point cloud
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            
            # Set camera parameters
            ctr = self.vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 3])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.8)
            
            return True
        except Exception as e:
            print(f"Failed to initialize 3D visualizer: {e}")
            return False
    
    def disparity_to_point_cloud(self, disparity: np.ndarray, left_image: np.ndarray, 
                                Q: np.ndarray, max_distance: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Convert disparity map to 3D point cloud"""
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Create mask for valid points
        mask = (disparity > 0) & (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < max_distance)
        
        # Extract valid points
        valid_points = points_3d[mask]
        
        # Get colors from left image
        if len(left_image.shape) == 3:
            colors = left_image[mask] / 255.0
        else:
            gray_colors = left_image[mask] / 255.0
            colors = np.stack([gray_colors, gray_colors, gray_colors], axis=1)
        
        # Limit number of points for performance
        if len(valid_points) > self.max_points:
            indices = np.random.choice(len(valid_points), self.max_points, replace=False)
            valid_points = valid_points[indices]
            colors = colors[indices]
        
        return valid_points, colors
    
    def update_point_cloud(self, disparity: np.ndarray, left_image: np.ndarray, 
                          Q: np.ndarray, detections: List = None):
        """Update point cloud with new data"""
        try:
            points, colors = self.disparity_to_point_cloud(disparity, left_image, Q)
            
            with self.lock:
                self.current_points = points
                self.current_colors = colors
                
        except Exception as e:
            print(f"Error updating point cloud: {e}")
    
    def add_detection_markers(self, detections: List, points: np.ndarray, colors: np.ndarray):
        """Add markers for detected objects in point cloud"""
        if not detections:
            return points, colors
        
        marker_points = []
        marker_colors = []
        
        for detection in detections:
            if 'distance' in detection and 'center' in detection:
                # Calculate approximate 3D position
                x = (detection['center'][0] - 320) * detection['distance'] / 500  # Rough approximation
                y = (detection['center'][1] - 240) * detection['distance'] / 500
                z = detection['distance']
                
                # Add marker point
                marker_points.append([x, y, z])
                
                # Color based on detection type
                if 'human' in detection.get('label', '').lower():
                    marker_colors.append([1.0, 0.0, 0.0])  # Red for humans
                else:
                    marker_colors.append([0.0, 1.0, 0.0])  # Green for other objects
        
        if marker_points:
            marker_points = np.array(marker_points)
            marker_colors = np.array(marker_colors)
            
            # Combine with existing points
            if len(points) > 0:
                points = np.vstack([points, marker_points])
                colors = np.vstack([colors, marker_colors])
            else:
                points = marker_points
                colors = marker_colors
        
        return points, colors
    
    def visualization_loop(self):
        """Main visualization loop (runs in separate thread)"""
        if not self.initialize_visualizer():
            return
        
        self.running = True
        
        while self.running:
            try:
                with self.lock:
                    if self.current_points is not None and len(self.current_points) > 0:
                        # Update point cloud
                        self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
                        self.pcd.colors = o3d.utility.Vector3dVector(self.current_colors)
                        
                        # Update visualization
                        self.vis.update_geometry(self.pcd)
                
                # Process events and render
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                break
        
        self.vis.destroy_window()
    
    def start(self):
        """Start visualization in separate thread"""
        if not self.running:
            viz_thread = Thread(target=self.visualization_loop, daemon=True)
            viz_thread.start()
    
    def stop(self):
        """Stop visualization"""
        self.running = False
    
    def save_point_cloud(self, filename: str):
        """Save current point cloud to file"""
        try:
            with self.lock:
                if self.current_points is not None:
                    pcd_save = o3d.geometry.PointCloud()
                    pcd_save.points = o3d.utility.Vector3dVector(self.current_points)
                    pcd_save.colors = o3d.utility.Vector3dVector(self.current_colors)
                    o3d.io.write_point_cloud(filename, pcd_save)
                    return True
        except Exception as e:
            print(f"Error saving point cloud: {e}")
        return False