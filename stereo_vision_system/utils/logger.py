"""
Advanced logging system for stereo vision system
Provides structured logging with performance monitoring and file rotation
"""

import logging
import logging.handlers
import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import traceback

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to the entire message
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        record.msg = f"{log_color}{record.msg}{reset_color}"
        
        return super().format(record)

class PerformanceMonitor:
    """Monitor system performance and log statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_times = []
        self.detection_times = []
        self.processing_times = []
        self.memory_usage = []
        self.lock = threading.Lock()
    
    def log_frame_time(self, frame_time: float):
        """Log frame processing time"""
        with self.lock:
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 1000:  # Keep last 1000 measurements
                self.frame_times.pop(0)
    
    def log_detection_time(self, detection_time: float):
        """Log AI detection time"""
        with self.lock:
            self.detection_times.append(detection_time)
            if len(self.detection_times) > 1000:
                self.detection_times.pop(0)
    
    def log_processing_time(self, processing_time: float):
        """Log stereo processing time"""
        with self.lock:
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 1000:
                self.processing_times.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {
                'uptime': time.time() - self.start_time,
                'frame_stats': self._calculate_stats(self.frame_times),
                'detection_stats': self._calculate_stats(self.detection_times),
                'processing_stats': self._calculate_stats(self.processing_times)
            }
            return stats
    
    def _calculate_stats(self, times: list) -> Dict[str, float]:
        """Calculate statistical metrics"""
        if not times:
            return {'avg': 0, 'min': 0, 'max': 0, 'fps': 0}
        
        avg_time = sum(times) / len(times)
        return {
            'avg': avg_time,
            'min': min(times),
            'max': max(times),
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }

class StereoVisionLogger:
    """Main logging system for stereo vision application"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('StereoVision')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Setup handlers
        self._setup_file_handlers(max_file_size, backup_count)
        if console_output:
            self._setup_console_handler()
        
        # Component loggers
        self.component_loggers = {}
        self._setup_component_loggers()
        
        self.logger.info("Stereo Vision Logger initialized")
    
    def _setup_file_handlers(self, max_file_size: int, backup_count: int):
        """Setup rotating file handlers"""
        
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "stereo_vision.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # Error log (errors and critical only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance log
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        
        # Create performance logger
        self.perf_logger = logging.getLogger('Performance')
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.propagate = False
    
    def _setup_console_handler(self):
        """Setup colored console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_component_loggers(self):
        """Setup specialized loggers for different components"""
        components = [
            'CameraManager', 'StereoProcessor', 'DetectorEngine', 
            'TrackerSystem', 'HUDOverlay', 'RadarMapper', 'TTSEngine'
        ]
        
        for component in components:
            logger = logging.getLogger(f'StereoVision.{component}')
            logger.setLevel(self.logger.level)
            self.component_loggers[component] = logger
    
    def get_component_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component"""
        return self.component_loggers.get(component, self.logger)
    
    def log_system_info(self):
        """Log system and environment information"""
        import platform
        import psutil
        
        self.logger.info("=== SYSTEM INFORMATION ===")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"CPU: {platform.processor()}")
        self.logger.info(f"CPU Cores: {psutil.cpu_count()}")
        self.logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU info (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.logger.info(f"GPU: {gpu.name} - {gpu.memoryTotal}MB VRAM")
        except ImportError:
            self.logger.warning("GPUtil not available - GPU info not logged")
        
        self.logger.info("=== SYSTEM INFORMATION END ===")
    
    def log_camera_config(self, config: Dict[str, Any]):
        """Log camera configuration"""
        self.logger.info("=== CAMERA CONFIGURATION ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=== CAMERA CONFIGURATION END ===")
    
    def log_detection_event(self, detections: list, distances: list):
        """Log detection events with distances"""
        if detections:
            detection_info = []
            for detection, distance in zip(detections, distances):
                detection_info.append(f"{detection['class']} at {distance:.2f}m")
            
            self.logger.info(f"Detections: {', '.join(detection_info)}")
    
    def log_performance_stats(self, force: bool = False):
        """Log performance statistics"""
        stats = self.perf_monitor.get_stats()
        
        # Log every 30 seconds or when forced
        if force or int(stats['uptime']) % 30 == 0:
            self.perf_logger.info(json.dumps(stats, indent=2))
            
            # Also log to main logger for important metrics
            frame_fps = stats['frame_stats']['fps']
            detection_fps = stats['detection_stats']['fps']
            
            if frame_fps < 10:  # Warning if FPS drops below 10
                self.logger.warning(f"Low frame rate: {frame_fps:.1f} FPS")
            
            if detection_fps < 5:  # Warning if detection FPS drops below 5
                self.logger.warning(f"Low detection rate: {detection_fps:.1f} FPS")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context and traceback"""
        self.logger.error(f"Exception occurred: {type(error).__name__}: {error}")
        
        if context:
            self.logger.error(f"Context: {json.dumps(context, indent=2)}")
        
        # Log full traceback
        self.logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    def create_session_log(self, session_id: str):
        """Create a separate log file for a specific session"""
        session_logger = logging.getLogger(f'Session.{session_id}')
        session_handler = logging.FileHandler(
            self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        session_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        session_handler.setFormatter(session_formatter)
        session_logger.addHandler(session_handler)
        session_logger.setLevel(logging.INFO)
        session_logger.propagate = False
        
        return session_logger

# Decorators for automatic logging
def log_performance(logger_instance: StereoVisionLogger, component: str):
    """Decorator to automatically log function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            component_logger = logger_instance.get_component_logger(component)
            
            try:
                component_logger.debug(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                component_logger.debug(f"Completed {func.__name__} in {execution_time:.4f}s")
                
                # Log to performance monitor
                if 'frame' in func.__name__.lower():
                    logger_instance.perf_monitor.log_frame_time(execution_time)
                elif 'detect' in func.__name__.lower():
                    logger_instance.perf_monitor.log_detection_time(execution_time)
                elif 'process' in func.__name__.lower():
                    logger_instance.perf_monitor.log_processing_time(execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger_instance.log_error_with_context(
                    e, 
                    {
                        'function': func.__name__,
                        'component': component,
                        'execution_time': execution_time,
                        'args': str(args)[:200],  # Truncate for safety
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        
        return wrapper
    return decorator

def log_method_calls(logger_instance: StereoVisionLogger, component: str):
    """Decorator to log method entry and exit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            component_logger = logger_instance.get_component_logger(component)
            component_logger.debug(f"Entering {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                component_logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                component_logger.error(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

# Global logger instance
_global_logger: Optional[StereoVisionLogger] = None

def get_logger() -> StereoVisionLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StereoVisionLogger()
    return _global_logger

def init_logger(log_dir: str = "logs", 
                log_level: str = "INFO",
                console_output: bool = True) -> StereoVisionLogger:
    """Initialize the global logger"""
    global _global_logger
    _global_logger = StereoVisionLogger(
        log_dir=log_dir,
        log_level=log_level,
        console_output=console_output
    )
    return _global_logger

# Convenience functions
def log_info(message: str, component: str = None):
    """Log info message"""
    logger = get_logger()
    if component:
        logger.get_component_logger(component).info(message)
    else:
        logger.logger.info(message)

def log_warning(message: str, component: str = None):
    """Log warning message"""
    logger = get_logger()
    if component:
        logger.get_component_logger(component).warning(message)
    else:
        logger.logger.warning(message)

def log_error(message: str, component: str = None):
    """Log error message"""
    logger = get_logger()
    if component:
        logger.get_component_logger(component).error(message)
    else:
        logger.logger.error(message)

def log_debug(message: str, component: str = None):
    """Log debug message"""
    logger = get_logger()
    if component:
        logger.get_component_logger(component).debug(message)
    else:
        logger.logger.debug(message)

# Example usage and testing
if __name__ == "__main__":
    # Initialize logger for testing
    logger = init_logger(log_level="DEBUG")
    
    # Log system info
    logger.log_system_info()
    
    # Test component logging
    cam_logger = logger.get_component_logger('CameraManager')
    cam_logger.info("Camera manager initialized")
    
    # Test performance logging
    @log_performance(logger, 'TestComponent')
    def test_function():
        time.sleep(0.1)  # Simulate processing time
        return "test result"
    
    result = test_function()
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.log_error_with_context(e, {'test_context': 'error_logging_test'})
    
    # Log performance stats
    logger.log_performance_stats(force=True)
    
    print("Logger testing completed. Check 'logs' directory for output files.")