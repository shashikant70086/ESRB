# core/tracker_system.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class TrackedObject:
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    distance: float
    confidence: float
    last_seen: float
    tracker: Optional[cv2.Tracker] = None
    trajectory: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []

class MultiObjectTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.disappeared = defaultdict(int)
        
    def _create_tracker(self) -> cv2.Tracker:
        """Create a new OpenCV tracker instance"""
        # Using CSRT for better accuracy
        return cv2.TrackerCSRT_create()
    
    def _calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point from bounding box"""
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detection dictionaries with keys:
                       'bbox', 'class_name', 'confidence', 'distance'
        
        Returns:
            List of tracked objects
        """
        current_time = time.time()
        
        # Convert detections to centers
        detection_centers = []
        for detection in detections:
            bbox = detection['bbox']
            center = self._get_center(bbox)
            detection_centers.append(center)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, detection in enumerate(detections):
                self._register_object(frame, detection, detection_centers[i], current_time)
        else:
            # Update existing trackers
            self._update_existing_trackers(frame, current_time)
            
            # Match detections to existing objects
            if len(detection_centers) > 0:
                self._match_detections_to_objects(
                    frame, detections, detection_centers, current_time
                )
        
        # Remove disappeared objects
        self._remove_disappeared_objects()
        
        return list(self.objects.values())
    
    def _update_existing_trackers(self, frame: np.ndarray, current_time: float):
        """Update existing object trackers"""
        objects_to_remove = []
        
        for object_id, obj in self.objects.items():
            if obj.tracker is not None:
                success, bbox = obj.tracker.update(frame)
                
                if success:
                    # Update object with new position
                    x, y, w, h = [int(v) for v in bbox]
                    obj.bbox = (x, y, w, h)
                    obj.center = self._get_center(obj.bbox)
                    obj.last_seen = current_time
                    obj.trajectory.append(obj.center)
                    
                    # Limit trajectory length
                    if len(obj.trajectory) > 50:
                        obj.trajectory = obj.trajectory[-50:]
                    
                    # Reset disappeared counter
                    self.disappeared[object_id] = 0
                else:
                    # Tracking failed, increment disappeared counter
                    self.disappeared[object_id] += 1
    
    def _match_detections_to_objects(self, frame: np.ndarray, detections: List[Dict], 
                                   detection_centers: List[Tuple[int, int]], current_time: float):
        """Match new detections to existing objects"""
        object_centers = [obj.center for obj in self.objects.values()]
        object_ids = list(self.objects.keys())
        
        if len(object_centers) > 0:
            # Compute distance matrix
            D = np.linalg.norm(
                np.array(object_centers)[:, np.newaxis] - np.array(detection_centers), 
                axis=2
            )
            
            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update matched objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    detection = detections[col]
                    
                    # Update object with new detection
                    self._update_object_with_detection(
                        frame, object_id, detection, detection_centers[col], current_time
                    )
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Register new objects for unmatched detections
            unused_detections = set(range(len(detections))) - used_col_indices
            for col in unused_detections:
                detection = detections[col]
                center = detection_centers[col]
                self._register_object(frame, detection, center, current_time)
            
            # Mark unmatched existing objects as disappeared
            unused_objects = set(range(len(object_ids))) - used_row_indices
            for row in unused_objects:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
    
    def _register_object(self, frame: np.ndarray, detection: Dict, 
                        center: Tuple[int, int], current_time: float):
        """Register a new object"""
        bbox = detection['bbox']
        
        # Create new tracker
        tracker = self._create_tracker()
        success = tracker.init(frame, bbox)
        
        if success:
            tracked_obj = TrackedObject(
                id=self.next_id,
                class_name=detection['class_name'],
                bbox=bbox,
                center=center,
                distance=detection.get('distance', 0.0),
                confidence=detection['confidence'],
                last_seen=current_time,
                tracker=tracker,
                trajectory=[center]
            )
            
            self.objects[self.next_id] = tracked_obj
            self.next_id += 1
    
    def _update_object_with_detection(self, frame: np.ndarray, object_id: int, 
                                    detection: Dict, center: Tuple[int, int], current_time: float):
        """Update existing object with new detection"""
        obj = self.objects[object_id]
        bbox = detection['bbox']
        
        # Update object properties
        obj.bbox = bbox
        obj.center = center
        obj.distance = detection.get('distance', obj.distance)
        obj.confidence = detection['confidence']
        obj.last_seen = current_time
        obj.trajectory.append(center)
        
        # Limit trajectory length
        if len(obj.trajectory) > 50:
            obj.trajectory = obj.trajectory[-50:]
        
        # Reinitialize tracker with new bbox
        obj.tracker = self._create_tracker()
        obj.tracker.init(frame, bbox)
        
        # Reset disappeared counter
        self.disappeared[object_id] = 0
    
    def _remove_disappeared_objects(self):
        """Remove objects that have disappeared for too long"""
        to_delete = []
        
        for object_id in list(self.disappeared.keys()):
            if self.disappeared[object_id] > self.max_disappeared:
                to_delete.append(object_id)
        
        for object_id in to_delete:
            del self.objects[object_id]
            del self.disappeared[object_id]
    
    def get_active_objects(self) -> List[TrackedObject]:
        """Get all currently active tracked objects"""
        return [obj for obj in self.objects.values() 
                if self.disappeared.get(obj.id, 0) < self.max_disappeared]
    
    def clear(self):
        """Clear all tracked objects"""
        self.objects.clear()
        self.disappeared.clear()
        self.next_id = 0