# audio/__init__.py
"""
Audio feedback module for stereo vision system.
Handles text-to-speech announcements and audio notifications.
"""

from .tts_engine import TTSEngine, VoiceConfig, AudioFeedback

__version__ = "1.0.0"
__all__ = [
    "TTSEngine",
    "VoiceConfig",
    "AudioFeedback"
]
