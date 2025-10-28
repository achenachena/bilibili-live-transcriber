"""
Configuration settings for the Bilibili live transcriber.
"""
import os

# Directory settings - relative to project root
# Get project root (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create directories if they don't exist
for directory in [VIDEO_DIR, AUDIO_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Whisper settings
WHISPER_MODEL = "large"  # Options: tiny, base, small, medium, large (large for best accuracy)
WHISPER_LANGUAGE = "zh"  # Chinese by default, can be set to None for auto-detection

# Audio processing settings
AUDIO_SAMPLE_RATE = 16000  # 16kHz for optimal processing
AUDIO_CHANNELS = 1  # Mono audio

# Diarization settings
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZATION_PIPELINE = "pyannote/speaker-diarization-3.1"

# Processing settings
MIN_SPEAKER_DURATION = 0.5  # Minimum duration for a speaker segment (seconds)
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration to split segments

# Output settings
OUTPUT_ENCODING = "utf-8"
