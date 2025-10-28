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
# Options: tiny, base, small, medium, large (large for best accuracy)
WHISPER_MODEL = "large"
WHISPER_LANGUAGE = "zh"  # Chinese by default, can be set to None for auto-detection

# Whisper anti-hallucination settings
# These parameters prevent repetitive output issues documented in:
# https://github.com/openai/whisper/discussions/2015
# https://github.com/openai/whisper/discussions/1962
# Key fix for repetitive hallucinations
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False
WHISPER_TEMPERATURE = 0.0  # Deterministic sampling to reduce hallucinations
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4  # Detect repetitive patterns
WHISPER_LOGPROB_THRESHOLD = -1.0  # Filter out low-confidence segments
WHISPER_NO_SPEECH_THRESHOLD = 0.6  # Better silence detection
WHISPER_FP16 = False  # Use FP32 to avoid CPU compatibility issues
WHISPER_WORD_TIMESTAMPS = False  # Use segment-level timestamps

# Repetitive segment filtering settings
# If same text appears 3+ times, consider it repetitive
REPETITIVE_SEGMENT_THRESHOLD = 3
MIN_SEGMENT_LENGTH = 3  # Minimum text length to keep (characters)

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

# Time formatting settings
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
HOURS_PER_DAY = 24

# Pipeline step settings
PIPELINE_STEPS = {
    "DOWNLOAD": 1,
    "EXTRACT": 2,
    "DIARIZE": 3,
    "TRANSCRIBE": 4,
    "MERGE": 5
}

# File processing settings
DEFAULT_SEGMENT_DURATION = 600  # 10 minutes in seconds
DEFAULT_SPLIT_THRESHOLD = 7200  # 2 hours in seconds
DEFAULT_SPLIT_THRESHOLD_CLI = 3600  # 1 hour in seconds (CLI default)
AUDIO_BITS_PER_SAMPLE = 16  # PCM 16-bit
MAX_TEXT_PREVIEW_LENGTH = 50  # Characters for text preview

# Audio codec settings
AUDIO_CODEC = "pcm_s16le"  # PCM 16-bit little endian
AUDIO_RESET_TIMESTAMPS = 1  # Reset timestamps flag

# File extension settings
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.webm', '.flv']

# Processing limits
MAX_SPEAKER_COUNT_DEFAULT = None  # No limit by default
MIN_OVERLAP_RATIO = 0.0  # Minimum overlap ratio for speaker detection
