# core/stereo_processor.py
"""
Stereo vision processing for depth estimation and 3D reconstruction
"""
import cv2
import numpy as np
import json
from typing import Tuple, Optional, Dict, List
from loguru import logger
from config.settings import StereoConfig, CameraConfig

class StereoVisionProcessor:
    """Complete stereo vision pipeline for depth estimation"""
    
    def __init__(self, camera_config: CameraConfig, stereo_config: StereoConfig):
        self.camera_config = camera_config
        self.stereo_config = stereo_config
        
        # Camera calibration matrices
        self.left_camera_matrix = None
        self.right_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_dist_coeffs = None
        self.rotation_matrix = None
        self.translation_vector = None
        
        # Rectification maps
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        
        # Stereo matcher
        self.stereo_matcher = None
        self.disparity_to_depth_map = None
        
        # Initialize stereo matcher
        self._initialize_stereo_matcher()
        
    def load_calibration(self, calibration_file: str) -> bool:
        """Load camera calibration parameters"""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            # Extract calibration parameters
            left_cam = calib_data['left_camera']
            right_cam = calib_data['right_camera']
            stereo_params = calib_data['stereo_params']
            
            self.left_camera_matrix = np.array(left_cam['camera_matrix'], dtype=np.float32)
            self.right_camera_matrix = np.array(right_cam['camera_matrix'], dtype=np.float32)
            self.left_dist_coeffs = np.array(left_cam['distortion_coeffs'], dtype=np.float32)
            self.right_dist_coeffs = np.array(right_cam['distortion_coeffs'], dtype=np.float32)
            
            self.rotation_matrix = np.array(stereo_params['rotation_matrix'], dtype=np.float32)
            self.translation_vector = np.array(stereo_params['translation_vector'], dtype=np.float32)
            
            # Compute rectification
            self._compute_rectification()
            
            logger.info("✓ Camera calibration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self._use_default_calibration()
            return False
    
    def _use_default_calibration(self):
        """Use default calibration parameters"""
        logger.info("Using default calibration parameters")
        
        # Default camera matrix (estimate based on resolution)
        h, w = self.camera_config.resolution
        fx = fy = self.camera_config.focal_length_px
        cx, cy = w // 2, h // 2
        
        self.left_camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.right_camera_matrix = self.left_camera_matrix.copy()
        
        # Minimal distortion
        self.left_dist_coeffs = np.zeros(5, dtype=np.float32)
        self.right_dist_coeffs = np.zeros(5, dtype=np.float32)
        
        # Identity rotation, baseline translation
        self.rotation_matrix = np.eye(3, dtype=np.float32)
        self.translation_vector = np.array([-self.camera_config.baseline_cm, 0, 0], dtype=np.float32)
        
        self._compute_rectification()
    
    def _compute_rectification(self):
        """Compute stereo rectification maps"""
        h, w = self.camera_config.resolution
        
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            (w, h), self.rotation_matrix, self.translation_vector,
            alpha=0.0
        )
        
        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1
        )
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1
        )
        
        # Store Q matrix for 3D reconstruction
        self.disparity_to_depth_map = Q
        
        logger.info("✓ Stereo rectification computed")
    
    def _initialize_stereo_matcher(self):
        """Initialize stereo matching algorithm"""
        if self.stereo_config.stereo_algorithm == "SGBM":
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=self.stereo_config.min_disparity,
                numDisparities=self.stereo_config.num_disparities,
                blockSize=self.stereo_config.block_size,
                uniquenessRatio=self.stereo_config.uniqueness_ratio,
                speckleWindowSize=self.stereo_config.speckle_window_size,
                speckleRange=self.stereo_config.speckle_range,
                disp12MaxDiff=self.stereo_config.disp12_max_diff,
                P1=8 * 3 * self.stereo_config.block_size ** 2,
                P2=32 * 3 * self.stereo_config.block_size ** 2
            )
        else:  # Block Matching
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=self.stereo_config.num_disparities,
                blockSize=self.stereo_config.block_size
            )
            
        logger.info(f"✓ Stereo matcher initialized: {self.stereo_config.stereo_algorithm}")
    
    def rectify_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify stereo frame pair"""
        if self.left_map1 is None or self.right_map1 is None:
            logger.warning("Rectification maps not computed, using original frames")
            return left_frame, right_frame
        
        left_rectified = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def compute_disparity(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """Compute disparity map from rectified stereo pair"""
        # Convert to grayscale if needed
        if len(left_frame.shape) == 3:
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_frame
            right_gray = right_frame
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity map to depth map (in cm)"""
        # Avoid division by zero
        valid_disparity = disparity > 0
        depth = np.zeros_like(disparity, dtype=np.float32)
        
        # Calculate depth using stereo formula: depth = (focal_length * baseline) / disparity
        focal_length = self.left_camera_matrix[0, 0]  # fx
        baseline = abs(self.translation_vector[0])  # baseline in cm
        
        depth[valid_disparity] = (focal_length * baseline) / disparity[valid_disparity]
        
        return depth
    
    def get_depth_at_point(self, disparity: np.ndarray, x: int, y: int, window_size: int = 5) -> float:
        """Get depth at specific point with averaging"""
        h, w = disparity.shape
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Define window
        half_window = window_size // 2
        x1, x2 = max(0, x - half_window), min(w, x + half_window + 1)
        y1, y2 = max(0, y - half_window), min(h, y + half_window + 1)
        
        # Get disparity window
        disparity_window = disparity[y1:y2, x1:x2]
        valid_disparities = disparity_window[disparity_window > 0]
        
        if len(valid_disparities) == 0:
            return 0.0
        
        # Calculate average depth
        avg_disparity = np.mean(valid_disparities)
        focal_length = self.left_camera_matrix[0, 0]
        baseline = abs(self.translation_vector[0])
        
        depth_cm = (focal_length * baseline) / avg_disparity
        return depth_cm
    
    def reconstruct_3d_points(self, disparity: np.ndarray) -> np.ndarray:
        """Reconstruct 3D points from disparity map"""
        if self.disparity_to_depth_map is None:
            logger.error("Q matrix not available for 3D reconstruction")
            return np.array([])
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.disparity_to_depth_map)
        
        # Filter valid points (remove points with infinite depth)
        mask = disparity > 0
        valid_points = points_3d[mask]
        
        # Remove outliers (points too far or too close)
        valid_mask = (valid_points[:, 2] > 10) & (valid_points[:, 2] < 1000)
        filtered_points = valid_points[valid_mask]
        
        return filtered_points
    
    def create_depth_colormap(self, depth: np.ndarray, max_depth: float = 500.0) -> np.ndarray:
        """Create colored depth visualization"""
        # Normalize depth to 0-255 range
        depth_normalized = np.clip(depth / max_depth * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Set invalid depths to black
        mask = depth <= 0
        depth_colored[mask] = [0, 0, 0]
        
        return depth_colored
    
    def process_stereo_frame(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Dict:
        """Complete stereo processing pipeline"""
        result = {
            'disparity': None,
            'depth': None,
            'depth_colored': None,
            'left_rectified': None,
            'right_rectified': None,
            'points_3d': None,
            'processing_time': 0.0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Rectify frames
            left_rect, right_rect = self.rectify_frames(left_frame, right_frame)
            result['left_rectified'] = left_rect
            result['right_rectified'] = right_rect
            
            # Compute disparity
            disparity = self.compute_disparity(left_rect, right_rect)
            result['disparity'] = disparity
            
            # Convert to depth
            depth = self.disparity_to_depth(disparity)
            result['depth'] = depth
            
            # Create colored depth map
            depth_colored = self.create_depth_colormap(depth)
            result['depth_colored'] = depth_colored
            
            # 3D reconstruction (optional, computationally expensive)
            # result['points_3d'] = self.reconstruct_3d_points(disparity)
            
            result['processing_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error in stereo processing: {e}")
        
        return result

class DepthEstimator:
    """Utility class for depth estimation from bounding boxes"""
    
    def __init__(self, stereo_processor: StereoVisionProcessor):
        self.stereo = stereo_processor
    
    def estimate_object_depth(self, disparity: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Estimate depth of object within bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Get center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get depth at center with averaging
        center_depth = self.stereo.get_depth_at_point(disparity, center_x, center_y, window_size=7)
        
        # Get depth statistics within bounding box
        bbox_disparity = disparity[y1:y2, x1:x2]
        valid_disparities = bbox_disparity[bbox_disparity > 0]
        
        if len(valid_disparities) == 0:
            return {
                'center_depth_cm': 0.0,
                'min_depth_cm': 0.0,
                'max_depth_cm': 0.0,
                'avg_depth_cm': 0.0,
                'confidence': 0.0
            }
        
        # Calculate depth statistics
        focal_length = self.stereo.left_camera_matrix[0, 0]
        baseline = abs(self.stereo.translation_vector[0])
        
        depths = (focal_length * baseline) / valid_disparities
        
        return {
            'center_depth_cm': center_depth,
            'min_depth_cm': float(np.min(depths)),
            'max_depth_cm': float(np.max(depths)),
            'avg_depth_cm': float(np.mean(depths)),
            'confidence': len(valid_disparities) / (bbox_disparity.size)  # Percentage of valid pixels
        }
    
    def get_distance_color(self, depth_cm: float, warning_distance: float = 500.0, danger_distance: float = 200.0) -> Tuple[int, int, int]:
        """Get color based on distance (BGR format)"""
        if depth_cm <= 0:
            return (128, 128, 128)  # Gray for invalid
        elif depth_cm < danger_distance:
            return (0, 0, 255)      # Red for danger
        elif depth_cm < warning_distance:
            return (0, 255, 255)    # Yellow for warning
        else:
            return (0, 255, 0)      # Green for safe

# Camera calibration utilities
class StereoCalibrator:
    """Stereo camera calibration helper"""
    
    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6)):
        self.chessboard_size = chessboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        self.objpoints = []  # 3d points in real world space
        self.imgpoints_left = []  # 2d points in left image plane
        self.imgpoints_right = []  # 2d points in right image plane
    
    def add_calibration_frame(self, left_frame: np.ndarray, right_frame: np.ndarray) -> bool:
        """Add a calibration frame pair"""
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)
        
        if ret_left and ret_right:
            # Refine corners
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
            
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            
            return True
        
        return False
    
    def calibrate(self, image_size: Tuple[int, int]) -> Dict:
        """Perform stereo calibration"""
        if len(self.objpoints) < 10:
            raise ValueError("Need at least 10 calibration frames")
        
        # Individual camera calibration
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None
        )
        
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None
        )
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            image_size, flags=flags, criteria=self.criteria
        )
        
        return {
            'left_camera': {
                'camera_matrix': mtx_left.tolist(),
                'distortion_coeffs': dist_left.tolist()
            },
            'right_camera': {
                'camera_matrix': mtx_right.tolist(),
                'distortion_coeffs': dist_right.tolist()
            },
            'stereo_params': {
                'rotation_matrix': R.tolist(),
                'translation_vector': T.tolist(),
                'essential_matrix': E.tolist(),
                'fundamental_matrix': F.tolist()
            },
            'calibration_error': ret_stereo
        }

if __name__ == "__main__":
    # Test stereo processor
    from config.settings import CameraConfig, StereoConfig
    
    camera_config = CameraConfig()
    stereo_config = StereoConfig()
    
    processor = StereoVisionProcessor(camera_config, stereo_config)
    processor._use_default_calibration()
    
    # Create test frames
    test_left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = processor.process_stereo_frame(test_left, test_right)
    print(f"Processing time: {result['processing_time']:.3f}s")
    print(f"Disparity shape: {result['disparity'].shape if result['disparity'] is not None else 'None'}")