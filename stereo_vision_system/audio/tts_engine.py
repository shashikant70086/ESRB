# audio/tts_engine.py
import pyttsx3
import threading
import queue
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from core.tracker_system import TrackedObject

@dataclass
class AudioMessage:
    text: str
    priority: int = 1  # 1=low, 2=medium, 3=high
    delay: float = 0.0  # Delay before speaking
    object_id: Optional[int] = None

class TTSEngine:
    def __init__(self):
        self.engine = None
        self.audio_queue = queue.PriorityQueue()
        self.audio_thread = None
        self.is_running = False
        self.last_announcements = {}  # Track last announcement time per object
        self.announcement_cooldown = 3.0  # Seconds between same object announcements
        
        # Audio settings
        self.voice_settings = {
            'rate': 180,      # Words per minute
            'volume': 0.8,    # Volume level (0.0 to 1.0)
            'voice_id': 0     # Voice selection
        }
        
        # Initialize TTS engine
        self._initialize_engine()
        
        # Start audio thread
        self._start_audio_thread()
    
    def _initialize_engine(self):
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set voice properties
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_settings['voice_id']:
                self.engine.setProperty('voice', voices[self.voice_settings['voice_id']].id)
            
            self.engine.setProperty('rate', self.voice_settings['rate'])
            self.engine.setProperty('volume', self.voice_settings['volume'])
            
            print("TTS Engine initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _start_audio_thread(self):
        """Start the audio processing thread"""
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
    
    def _audio_worker(self):
        """Audio processing worker thread"""
        while self.is_running:
            try:
                # Get message from queue (blocking with timeout)
                priority, timestamp, message = self.audio_queue.get(timeout=1.0)
                
                # Apply delay if specified
                if message.delay > 0:
                    time.sleep(message.delay)
                
                # Speak the message
                self._speak_text(message.text)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio worker error: {e}")
    
    def _speak_text(self, text: str):
        """Speak the given text"""
        if self.engine is None:
            print(f"TTS not available, would say: {text}")
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    
    def _should_announce_object(self, obj: TrackedObject) -> bool:
        """Check if object should be announced based on cooldown"""
        current_time = time.time()
        
        if obj.id not in self.last_announcements:
            self.last_announcements[obj.id] = current_time
            return True
        
        time_since_last = current_time - self.last_announcements[obj.id]
        if time_since_last >= self.announcement_cooldown:
            self.last_announcements[obj.id] = current_time
            return True
        
        return False
    
    def _generate_detection_message(self, obj: TrackedObject) -> str:
        """Generate detection announcement message"""
        class_name = obj.class_name.lower()
        distance = obj.distance
        
        # Customize message based on object type and distance
        if class_name in ['human', 'person']:
            if distance < 2.0:
                return f"Warning! Human detected at {distance:.1f} meters. Very close proximity."
            elif distance < 5.0:
                return f"Human detected at {distance:.1f} meters."
            else:
                return f"Human contact at {distance:.1f} meters."
        
        elif class_name in ['animal', 'dog', 'cat', 'bird']:
            return f"{class_name.capitalize()} detected at {distance:.1f} meters."
        
        elif class_name in ['car', 'truck', 'motorcycle', 'vehicle']:
            return f"Vehicle detected at {distance:.1f} meters."
        
        else:
            return f"Object detected at {distance:.1f} meters."
    
    def _get_threat_assessment(self, objects: List[TrackedObject]) -> str:
        """Generate threat assessment message"""
        if not objects:
            return "Area clear. No objects detected."
        
        human_count = sum(1 for obj in objects if obj.class_name.lower() in ['human', 'person'])
        animal_count = sum(1 for obj in objects if obj.class_name.lower() in ['animal', 'dog', 'cat', 'bird'])
        vehicle_count = sum(1 for obj in objects if obj.class_name.lower() in ['car', 'truck', 'motorcycle', 'vehicle'])
        
        # Check for close proximity threats
        close_threats = [obj for obj in objects if obj.distance < 3.0 and obj.class_name.lower() in ['human', 'person']]
        
        if close_threats:
            return f"Alert! {len(close_threats)} close proximity contact{'s' if len(close_threats) > 1 else ''}."
        
        # General status
        status_parts = []
        if human_count > 0:
            status_parts.append(f"{human_count} human{'s' if human_count > 1 else ''}")
        if animal_count > 0:
            status_parts.append(f"{animal_count} animal{'s' if animal_count > 1 else ''}")
        if vehicle_count > 0:
            status_parts.append(f"{vehicle_count} vehicle{'s' if vehicle_count > 1 else ''}")
        
        if status_parts:
            return f"Monitoring {', '.join(status_parts)} in area."
        else:
            return f"Tracking {len(objects)} object{'s' if len(objects) > 1 else ''}."
    
    def announce_detection(self, obj: TrackedObject, priority: int = 2):
        """Announce object detection"""
        if not self._should_announce_object(obj):
            return
        
        message_text = self._generate_detection_message(obj)
        message = AudioMessage(
            text=message_text,
            priority=priority,
            object_id=obj.id
        )
        
        # Add to queue with priority and timestamp
        self.audio_queue.put((priority, time.time(), message))
    
    def announce_status(self, objects: List[TrackedObject], priority: int = 1):
        """Announce general status"""
        message_text = self._get_threat_assessment(objects)
        message = AudioMessage(
            text=message_text,
            priority=priority
        )
        
        self.audio_queue.put((priority, time.time(), message))
    
    def announce_custom(self, text: str, priority: int = 2, delay: float = 0.0):
        """Announce custom message"""
        message = AudioMessage(
            text=text,
            priority=priority,
            delay=delay
        )
        
        self.audio_queue.put((priority, time.time(), message))
    
    def announce_system_status(self, fps: float, object_count: int):
        """Announce system status"""
        if fps < 10:
            self.announce_custom("Warning: Low frame rate detected. Performance degraded.", priority=3)
        
        if object_count > 10:
            self.announce_custom(f"High activity detected. Tracking {object_count} objects.", priority=2)
    
    def set_voice_settings(self, rate: Optional[int] = None, volume: Optional[float] = None, 
                          voice_id: Optional[int] = None):
        """Update voice settings"""
        if self.engine is None:
            return
        
        try:
            if rate is not None:
                self.voice_settings['rate'] = max(50, min(300, rate))
                self.engine.setProperty('rate', self.voice_settings['rate'])
            
            if volume is not None:
                self.voice_settings['volume'] = max(0.0, min(1.0, volume))
                self.engine.setProperty('volume', self.voice_settings['volume'])
            
            if voice_id is not None:
                voices = self.engine.getProperty('voices')
                if voices and 0 <= voice_id < len(voices):
                    self.voice_settings['voice_id'] = voice_id
                    self.engine.setProperty('voice', voices[voice_id].id)
            
        except Exception as e:
            print(f"Error setting voice properties: {e}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        if self.engine is None:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [voice.name for voice in voices] if voices else []
        except:
            return []
    
    def set_announcement_cooldown(self, seconds: float):
        """Set cooldown period between announcements for same object"""
        self.announcement_cooldown = max(0.5, seconds)
    
    def clear_queue(self):
        """Clear all pending audio messages"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    def is_speaking(self) -> bool:
        """Check if TTS engine is currently speaking"""
        if self.engine is None:
            return False
        
        try:
            return self.engine.isBusy()
        except:
            return False
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
    
    def shutdown(self):
        """Shutdown the TTS engine"""
        self.is_running = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
        
        print("TTS Engine shutdown complete")
    
    def get_queue_size(self) -> int:
        """Get current audio queue size"""
        return self.audio_queue.qsize()
    
    def get_engine_info(self) -> Dict:
        """Get TTS engine information"""
        info = {
            'engine_available': self.engine is not None,
            'is_running': self.is_running,
            'queue_size': self.get_queue_size(),
            'is_speaking': self.is_speaking(),
            'settings': self.voice_settings.copy(),
            'available_voices': self.get_available_voices()
        }
        
        return info