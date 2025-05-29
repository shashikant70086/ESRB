# core/detector_engine.py
"""
AI detection engine using MediaPipe and YOLOv8 for human/animal detection
"""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
from loguru import logger
from config.settings import DetectionConfig

class Detection:
    """Detection result container"""
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, 
                 class_id: int, class_name: str, track_id: Optional[int] = None):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id
        self.depth_info = None
        
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class MediaPipeDetector:
    """MediaPipe-based human pose detection"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        logger.info("✓ MediaPipe detector initialized")
    
    def detect_humans(self, frame: np.ndarray) -> List[Detection]:
        """Detect humans using pose estimation"""
        detections = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with holistic model
        results = self.holistic.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract bounding box from pose landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get all landmark coordinates
            x_coords = [lm.x * frame.shape[1] for lm in landmarks]
            y_coords = [lm.y * frame.shape[0] for lm in landmarks]
            
            # Filter out landmarks with low visibility
            valid_coords = [(x, y) for x, y, lm in zip(x_coords, y_coords, landmarks) 
                           if lm.visibility > 0.5]
            
            if len(valid_coords) > 5:  # Need minimum landmarks for valid detection
                x_coords, y_coords = zip(*valid_coords)
                
                # Calculate bounding box with padding
                padding = 20
                x1 = max(0, int(min(x_coords)) - padding)
                y1 = max(0, int(min(y_coords)) - padding)
                x2 = min(frame.shape[1], int(max(x_coords)) + padding)
                y2 = min(frame.shape[0], int(max(y_coords)) + padding)
                
                # Calculate confidence based on landmark visibility
                confidence = np.mean([lm.visibility for lm in landmarks])
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=0,
                    class_name="person"
                )
                detections.append(detection)
        
        return detections
    
    def get_pose_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """Get detailed pose keypoints"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x * frame.shape[1],
                    'y': landmark.y * frame.shape[0],
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            return {
                'pose_keypoints': keypoints,
                'face_landmarks': results.face_landmarks,
                'left_hand_landmarks': results.left_hand_landmarks,
                'right_hand_landmarks': results.right_hand_landmarks
            }
        
        return None

class YOLODetector:
    """YOLOv8-based object detection"""
    
    # COCO class names for humans and animals
    HUMAN_ANIMAL_CLASSES = {
        0: 'person',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
    }
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        try:
            self.model = YOLO(config.yolo_model)
            if config.use_tensorrt and torch.cuda.is_available():
                # Export to TensorRT for faster inference
                self.model.export(format='engine', device=0)
            
            logger.info(f"✓ YOLO detector initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using YOLO"""
        if self.model is None:
            return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.config.confidence_threshold, device=self.device)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Filter for humans and animals only
                        if class_id in self.HUMAN_ANIMAL_CLASSES:
                            class_name = self.HUMAN_ANIMAL_CLASSES[class_id]
                            
                            detection = Detection(
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            )
                            detections.append(detection)
            
            # Apply NMS to remove overlapping detections
            detections = self._apply_nms(detections)
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [list(det.bbox) for det in detections]
        confidences = [det.confidence for det in detections]
        class_ids = [det.class_id for det in detections]
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.config.confidence_threshold, 
            self.config.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []

class MultiModalDetector:
    """Combined MediaPipe + YOLO detection system"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        # Initialize detectors
        self.mediapipe_detector = MediaPipeDetector() if config.enable_mediapipe else None
        self.yolo_detector = YOLODetector(config) if config.enable_yolo else None
        
        # Detection fusion parameters
        self.overlap_threshold = 0.5  # IoU threshold for merging detections
        
        logger.info("✓ Multi-modal detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection using all enabled detectors"""
        all_detections = []
        
        # MediaPipe detection
        if self.mediapipe_detector:
            mp_detections = self.mediapipe_detector.detect_humans(frame)
            all_detections.extend(mp_detections)
        
        # YOLO detection
        if self.yolo_detector:
            yolo_detections = self.yolo_detector.detect_objects(frame)
            all_detections.extend(yolo_detections)
        
        # Fuse detections (remove duplicates, merge overlapping)
        fused_detections = self._fuse_detections(all_detections)
        
        # Limit to max detections
        if len(fused_detections) > self.config.max_detections:
            # Sort by confidence and take top N
            fused_detections.sort(key=lambda x: x.confidence, reverse=True)
            fused_detections = fused_detections[:self.config.max_detections]
        
        return fused_detections
    
    def _fuse_detections(self, detections: List[Detection]) -> List[Detection]:
        """Fuse overlapping detections from different sources"""
        if len(detections) <= 1:
            return detections
        
        fused = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find overlapping detections
            overlapping = [det1]
            overlapping_indices = [i]
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1.bbox, det2.bbox)
                if iou > self.overlap_threshold:
                    overlapping.append(det2)
                    overlapping_indices.append(j)
            
            # Merge overlapping detections
            if len(overlapping) > 1:
                merged = self._merge_detections(overlapping)
                fused.append(merged)
            else:
                fused.append(det1)
            
            # Mark as used
            used.update(overlapping_indices)
        
        return fused
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_detections(self, detections: List[Detection]) -> Detection:
        """Merge multiple overlapping detections"""
        # Use detection with highest confidence as base
        best_detection = max(detections, key=lambda x: x.confidence)
        
        # Average bounding boxes weighted by confidence
        total_weight = sum(det.confidence for det in detections)
        
        x1 = sum(det.bbox[0] * det.confidence for det in detections) / total_weight
        y1 = sum(det.bbox[1] * det.confidence for det in detections) / total_weight
        x2 = sum(det.bbox[2] * det.confidence for det in detections) / total_weight
        y2 = sum(det.bbox[3] * det.confidence for det in detections) / total_weight
        
        # Average confidence
        avg_confidence = sum(det.confidence for det in detections) / len(detections)
        
        return Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=avg_confidence,
            class_id=best_detection.class_id,
            class_name=best_detection.class_name
        )

# Performance monitoring
class DetectionProfiler:
    """Profile detection performance"""
    
    def __init__(self):
        self.detection_times = []
        self.fps_history = []
        self.last_time = None
    
    def start_frame(self):
        """Start timing a frame"""
        import time
        self.last_time = time.time()
    
    def end_frame(self) -> float:
        """End timing and return FPS"""
        if self.last_time is None:
            return 0.0
        
        import time
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.detection_times.append(frame_time)
        
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.fps_history.append(fps)
        
        # Keep only recent history
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-50:]
            self.fps_history = self.fps_history[-50:]
        
        return fps
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.detection_times:
            return {'avg_fps': 0.0, 'avg_detection_time': 0.0}
        
        return {
            'avg_fps': np.mean(self.fps_history),
            'avg_detection_time': np.mean(self.detection_times),
            'min_fps': np.min(self.fps_history),
            'max_fps': np.max(self.fps_history)
        }

if __name__ == "__main__":
    # Test detection system
    from config.settings import DetectionConfig
    
    config = DetectionConfig()
    detector = MultiModalDetector(config)
    profiler = DetectionProfiler()
    
    # Test with dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    profiler.start_frame()
    detections = detector.detect(test_frame)
    fps = profiler.end_frame()
    
    print(f"Detected {len(detections)} objects")
    print(f"FPS: {fps:.1f}")
    print(f"Stats: {profiler.get_stats()}")