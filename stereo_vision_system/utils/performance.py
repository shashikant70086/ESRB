# utils/performance.py
import threading
import time
import psutil
import numpy as np
from collections import deque
from threading import Thread, Lock, Event
import queue
import cv2

class PerformanceMonitor:
    """Monitor system performance and FPS"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
        self.lock = Lock()
        
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        with self.lock:
            if current_time - self.last_time >= 1.0:
                fps = self.frame_count / (current_time - self.last_time)
                self.fps_history.append(fps)
                self.frame_count = 0
                self.last_time = current_time
            else:
                self.frame_count += 1
    
    def update_system_stats(self):
        """Update CPU and memory usage"""
        with self.lock:
            self.cpu_history.append(psutil.cpu_percent())
            self.memory_history.append(psutil.virtual_memory().percent)
    
    def get_average_fps(self) -> float:
        """Get average FPS over window"""
        with self.lock:
            return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def get_current_stats(self) -> dict:
        """Get current performance statistics"""
        with self.lock:
            return {
                'fps': self.get_average_fps(),
                'cpu_percent': np.mean(self.cpu_history) if self.cpu_history else 0.0,
                'memory_percent': np.mean(self.memory_history) if self.memory_history else 0.0,
                'frame_count': len(self.fps_history)
            }

class ThreadSafeQueue:
    """Thread-safe queue for frame processing"""
    
    def __init__(self, maxsize: int = 10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = Lock()
    
    def put(self, item, block: bool = True, timeout: float = None):
        """Add item to queue"""
        try:
            self.queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, block: bool = True, timeout: float = None):
        """Get item from queue"""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

class FrameProcessor:
    """Multi-threaded frame processing pipeline"""
    
    def __init__(self, max_queue_size: int = 5):
        self.input_queue = ThreadSafeQueue(max_queue_size)
        self.output_queue = ThreadSafeQueue(max_queue_size)
        self.processing_thread = None
        self.running = False
        self.process_func = None
        
    def set_processing_function(self, func):
        """Set the processing function"""
        self.process_func = func
    
    def start_processing(self):
        """Start processing thread"""
        if not self.running and self.process_func:
            self.running = True
            self.processing_thread = Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
    
    def stop_processing(self):
        """Stop processing thread"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get frame from input queue
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                # Process frame
                result = self.process_func(frame_data)
                
                # Put result in output queue
                if result is not None:
                    self.output_queue.put(result, block=False)
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue
    
    def add_frame(self, frame_data):
        """Add frame for processing"""
        return self.input_queue.put(frame_data, block=False)
    
    def get_result(self):
        """Get processed result"""
        return self.output_queue.get(block=False)

class GPUMemoryManager:
    """Manage GPU memory for optimal performance"""
    
    def __init__(self):
        self.allocated_memory = 0
        self.max_memory = self._get_gpu_memory()
        
    def _get_gpu_memory(self) -> int:
        """Get available GPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass
        return 4 * 1024 * 1024 * 1024  # Default 4GB
    
    def optimize_opencv_for_gpu(self):
        """Optimize OpenCV for GPU usage"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
                return True
        except:
            pass
        return False
    
    def clear_gpu_cache(self):
        """Clear GPU cache if using PyTorch"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

class AdaptiveFrameSkipper:
    """Dynamically skip frames to maintain target FPS"""
    
    def __init__(self, target_fps: float = 20.0, adaptation_rate: float = 0.1):
        self.target_fps = target_fps
        self.adaptation_rate = adaptation_rate
        self.target_frame_time = 1.0 / target_fps
        self.skip_ratio = 0.0
        self.frame_times = deque(maxlen=10)
        self.frame_counter = 0
        
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed"""
        self.frame_counter += 1
        
        # Always process first frame
        if self.frame_counter == 1:
            return True
        
        # Calculate skip pattern based on current skip ratio
        skip_every_n = max(1, int(1.0 / (1.0 - self.skip_ratio))) if self.skip_ratio < 1.0 else 2
        
        return (self.frame_counter % skip_every_n) == 0
    
    def update_timing(self, processing_time: float):
        """Update timing information and adjust skip ratio"""
        self.frame_times.append(processing_time)
        
        if len(self.frame_times) >= 5:
            avg_time = np.mean(self.frame_times)
            
            if avg_time > self.target_frame_time:
                # Too slow, increase skip ratio
                self.skip_ratio = min(0.8, self.skip_ratio + self.adaptation_rate)
            elif avg_time < self.target_frame_time * 0.8:
                # Fast enough, decrease skip ratio
                self.skip_ratio = max(0.0, self.skip_ratio - self.adaptation_rate)
    
    def get_effective_fps(self) -> float:
        """Get effective FPS considering skipping"""
        if len(self.frame_times) > 0:
            avg_time = np.mean(self.frame_times)
            return (1.0 - self.skip_ratio) / avg_time
        return self.target_fps

class ResourceOptimizer:
    """Optimize system resources for real-time processing"""
    
    def __init__(self):
        self.original_priority = None
        
    def set_high_priority(self):
        """Set high priority for current process"""
        try:
            import os
            if os.name == 'nt':  # Windows
                import psutil
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:  # Unix-like
                os.nice(-10)
            return True
        except Exception as e:
            print(f"Could not set high priority: {e}")
            return False
    
    def optimize_opencv_threads(self, num_threads: int = None):
        """Optimize OpenCV threading"""
        if num_threads is None:
            num_threads = min(4, psutil.cpu_count())
        
        cv2.setNumThreads(num_threads)
        cv2.setUseOptimized(True)
        
        # Enable OpenCL if available
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    
    def get_optimal_batch_size(self, available_memory_gb: float = 4.0) -> int:
        """Calculate optimal batch size for processing"""
        # Rough estimation based on available memory
        if available_memory_gb >= 8:
            return 8
        elif available_memory_gb >= 4:
            return 4
        else:
            return 2