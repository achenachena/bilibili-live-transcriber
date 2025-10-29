"""
Configuration settings for the Bilibili live transcriber.
"""
import os
from typing import Optional

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

# Download optimization settings
# Audio-optimized format: best audio quality + worst video quality
# This significantly reduces download size and time since we only need audio
DOWNLOAD_FORMAT_AUDIO_OPTIMIZED = 'worstvideo+bestaudio/worst+bestaudio/bestaudio'
# Fallback format if audio-only not available
DOWNLOAD_FORMAT_FALLBACK = 'best[ext=mp4]/best'
# Output format for merged files
DOWNLOAD_MERGE_FORMAT = 'mp4'

# Parallel processing settings
# Maximum number of parallel workers for processing
# Set to None to use all available CPU cores
MAX_PARALLEL_WORKERS = None  # None = use all CPU cores
# Minimum number of segments/files to enable parallel processing
MIN_PARALLEL_THRESHOLD = 2  # Only parallelize if 2+ items
# Maximum number of parallel workers for batch processing
MAX_BATCH_WORKERS = 4  # Limit batch processing to avoid overwhelming system

# Punctuation restoration settings
PUNCTUATION_RULES = {
    'zh': {
        'sentence_endings': ['。', '！', '？'],
        'pause_indicators': ['，', '、', '；'],
        'question_words': ['什么', '怎么', '为什么', '哪里', '谁', '哪个', '多少'],
        'exclamation_words': ['好', '太', '真', '确实', '当然', '绝对']
    },
    'en': {
        'sentence_endings': ['.', '!', '?'],
        'pause_indicators': [',', ';'],
        'question_words': ['what', 'how', 'why', 'where', 'who', 'which', 'when'],
        'exclamation_words': ['great', 'amazing', 'wow', 'yes', 'no', 'sure']
    }
}

# Processing limits
MAX_SPEAKER_COUNT_DEFAULT = None  # No limit by default
MIN_OVERLAP_RATIO = 0.0  # Minimum overlap ratio for speaker detection

# Model caching for performance optimization
_MODEL_CACHE = {}


def get_cached_whisper_model(model_size: str):
    """
    Get cached Whisper model or load if not cached.

    This significantly improves performance by avoiding model reloading
    for each transcription. Models are cached globally across the session.
    """
    if model_size not in _MODEL_CACHE:
        try:
            import whisper  # pylint: disable=import-outside-toplevel
            _MODEL_CACHE[model_size] = whisper.load_model(model_size)
        except ImportError as e:
            raise ImportError(f"Whisper module not available: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model {model_size}: {e}") from e

    return _MODEL_CACHE[model_size]


def get_cached_diarization_pipeline():
    """
    Get cached diarization pipeline or load if not cached.

    This significantly improves performance by avoiding pipeline reloading
    for each audio file. Pipeline is cached globally across the session.
    """
    if 'diarization_pipeline' not in _MODEL_CACHE:
        try:
            from pyannote.audio import Pipeline  # pylint: disable=import-outside-toplevel

            # Get HuggingFace token from environment
            hf_token = os.environ.get(
                'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

            if hf_token:
                _MODEL_CACHE['diarization_pipeline'] = Pipeline.from_pretrained(
                    DIARIZATION_MODEL,
                    use_auth_token=hf_token
                )
            else:
                _MODEL_CACHE['diarization_pipeline'] = Pipeline.from_pretrained(
                    DIARIZATION_MODEL)

        except ImportError as e:
            raise ImportError(f"pyannote.audio not available: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization pipeline: {e}") from e

    return _MODEL_CACHE['diarization_pipeline']


def cleanup_model_cache():
    """
    Clean up model cache to free memory.

    Useful for long-running processes or when memory is limited.
    """
    for key in list(_MODEL_CACHE.keys()):
        if hasattr(_MODEL_CACHE[key], 'cpu'):
            _MODEL_CACHE[key] = _MODEL_CACHE[key].cpu()

    import gc
    gc.collect()


def get_optimal_workers(num_items: int,
                        max_workers: Optional[int] = None) -> int:
    """
    Calculate optimal number of workers for parallel processing.

    Args:
        num_items: Number of items to process
        max_workers: Maximum workers to use (None = use all CPU cores)

    Returns:
        Optimal number of workers
    """
    import multiprocessing

    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Use provided max_workers or default from config
    if max_workers is None:
        max_workers = MAX_PARALLEL_WORKERS or cpu_count

    # Don't exceed the number of items
    optimal_workers = min(num_items, max_workers, cpu_count)

    # Don't parallelize if below threshold
    if num_items < MIN_PARALLEL_THRESHOLD:
        return 1

    return max(1, optimal_workers)


def should_use_parallel_processing(num_items: int) -> bool:
    """
    Determine if parallel processing should be used.

    Args:
        num_items: Number of items to process

    Returns:
        True if parallel processing should be used
    """
    return num_items >= MIN_PARALLEL_THRESHOLD
