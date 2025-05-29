# core/camera_manager.py
"""
ESP32-CAM dual stream manager with synchronization and frame processing
"""
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple, Dict
from loguru import logger
from config.settings import CameraConfig

class ESP32CameraStream:
    """Individual ESP32-CAM stream handler"""
    
    def __init__(self, ip_address: str, camera_id: str, rotation: int = 0, mirror: bool = False):
        self.ip_address = ip_address
        self.camera_id = camera_id
        self.rotation = rotation
        self.mirror = mirror
        self.stream_url = f"http://{ip_address}/stream"
        
        self.cap = None
        self.frame_queue = Queue(maxsize=5)
        self.is_running = False
        self.thread = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def start(self) -> bool:
        """Start camera stream capture"""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.stream_url}")
                return False
                
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"✓ Camera {self.camera_id} started: {self.ip_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {self.camera_id}: {e}")
            return False
    
    def stop(self):
        """Stop camera stream"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {self.camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Apply transformations
                processed_frame = self._process_frame(frame)
                
                # Update FPS counter
                self._update_fps()
                
                # Add to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put({
                        'frame': processed_frame,
                        'timestamp': time.time(),
                        'fps': self.current_fps
                    })
                else:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put({
                            'frame': processed_frame,
                            'timestamp': time.time(),
                            'fps': self.current_fps
                        })
                    except Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in capture loop for {self.camera_id}: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation and mirror transformations"""
        # Apply rotation
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply mirror
        if self.mirror:
            frame = cv2.flip(frame, 1)
            
        return frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_latest_frame(self) -> Optional[tuple]:
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def is_healthy(self) -> bool:
        """Check if camera stream is healthy"""
        return self.is_running and self.cap and self.cap.isOpened()

class DualCameraManager:
    """Manager for synchronized dual ESP32-CAM setup"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.left_camera = ESP32CameraStream(
            config.left_ip, 
            "LEFT", 
            config.left_rotation, 
            config.left_mirror
        )
        self.right_camera = ESP32CameraStream(
            config.right_ip, 
            "RIGHT", 
            config.right_rotation, 
            config.right_mirror
        )
        
        self.sync_threshold = 0.1  # Max time difference for sync (seconds)
        self.frame_count = 0
        self.dropped_frames = 0
        
    def start(self) -> bool:
        """Start both camera streams"""
        logger.info("Starting dual camera system...")
        
        left_ok = self.left_camera.start()
        right_ok = self.right_camera.start()
        
        if left_ok and right_ok:
            logger.info("✓ Dual camera system started successfully")
            # Wait for streams to stabilize
            time.sleep(2.0)
            return True
        else:
            logger.error("Failed to start dual camera system")
            self.stop()
            return False
    
    def stop(self):
        """Stop both camera streams"""
        logger.info("Stopping dual camera system...")
        self.left_camera.stop()
        self.right_camera.stop()
        logger.info("✓ Dual camera system stopped")
    
    def get_synchronized_frames(self) -> Optional[Dict]:
        """Get synchronized frames from both cameras"""
        left_data = self.left_camera.get_latest_frame()
        right_data = self.right_camera.get_latest_frame()
        
        if not left_data or not right_data:
            return None
        
        # Check synchronization
        time_diff = abs(left_data['timestamp'] - right_data['timestamp'])
        if time_diff > self.sync_threshold:
            self.dropped_frames += 1
            logger.debug(f"Frames not synchronized: {time_diff:.3f}s difference")
            return None
        
        self.frame_count += 1
        
        return {
            'left_frame': left_data['frame'],
            'right_frame': right_data['frame'],
            'timestamp': (left_data['timestamp'] + right_data['timestamp']) / 2,
            'left_fps': left_data['fps'],
            'right_fps': right_data['fps'],
            'sync_quality': 1.0 - (time_diff / self.sync_threshold)
        }
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        total_frames = self.frame_count + self.dropped_frames
        sync_rate = (self.frame_count / total_frames * 100) if total_frames > 0 else 0
        
        return {
            'total_frames_processed': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'synchronization_rate': sync_rate,
            'left_camera_fps': self.left_camera.current_fps,
            'right_camera_fps': self.right_camera.current_fps,
            'left_camera_healthy': self.left_camera.is_healthy(),
            'right_camera_healthy': self.right_camera.is_healthy()
        }
    
    def is_system_healthy(self) -> bool:
        """Check if both cameras are functioning"""
        return (self.left_camera.is_healthy() and 
                self.right_camera.is_healthy())

# Utility functions for camera testing
def test_camera_connection(ip_address: str) -> bool:
    """Test if ESP32-CAM is accessible"""
    try:
        import requests
        response = requests.get(f"http://{ip_address}", timeout=5)
        return response.status_code == 200
    except:
        return False

def discover_esp32_cameras(ip_range: str = "192.168.1") -> List[str]:
    """Discover ESP32-CAM devices on network"""
    import concurrent.futures
    import requests
    
    def check_ip(ip):
        try:
            response = requests.get(f"http://{ip}", timeout=2)
            if "ESP32-CAM" in response.text or response.status_code == 200:
                return ip
        except:
            pass
        return None
    
    ips = [f"{ip_range}.{i}" for i in range(100, 200)]
    found_cameras = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(check_ip, ips)
        found_cameras = [ip for ip in results if ip is not None]
    
    return found_cameras

if __name__ == "__main__":
    # Test camera discovery
    print("Discovering ESP32-CAM devices...")
    cameras = discover_esp32_cameras()
    print(f"Found cameras: {cameras}")
    
    if len(cameras) >= 2:
        from config.settings import CameraConfig
        config = CameraConfig(left_ip=cameras[0], right_ip=cameras[1])
        
        manager = DualCameraManager(config)
        if manager.start():
            try:
                for i in range(50):  # Test for 50 frames
                    frames = manager.get_synchronized_frames()
                    if frames:
                        print(f"Frame {i}: {frames['left_frame'].shape}, sync: {frames['sync_quality']:.2f}")
                    time.sleep(0.1)
            finally:
                manager.stop()
    else:
        print("Need at least 2 ESP32-CAM devices for stereo vision")