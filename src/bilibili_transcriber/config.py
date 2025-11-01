"""
Configuration settings for the Bilibili live transcriber.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

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
WHISPER_FP16 = False  # Use FP32 on CPU, will be enabled on GPU
WHISPER_WORD_TIMESTAMPS = False  # Use segment-level timestamps

# GPU acceleration settings
# Enable GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
ENABLE_GPU = True  # Set to False to force CPU
# Use Core ML on Apple Silicon (experimental, requires coremltools)
USE_CORE_ML = True
CORE_ML_CACHE_DIR = os.path.join(PROJECT_ROOT,
                                 ".coreml_cache")  # Cache converted models
WHISPER_BEAM_SIZE = 5  # Beam search size for decoding (lower = faster)
WHISPER_BEST_OF = 1  # Number of candidates (1 = greedy search, faster)

# Voice Activity Detection (VAD) settings
ENABLE_VAD = True  # Enable VAD pre-filtering to skip silence
VAD_MIN_SILENCE_DURATION_MS = 500  # Minimum silence to split segments (ms)
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum speech to keep segment (ms)

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
_DEVICE_CACHE = None  # pylint: disable=invalid-name
_CORE_ML_AVAILABLE = None  # pylint: disable=invalid-name


def get_device():
    """
    Detect and return the best available device for inference.

    Priority:
    1. MPS (Metal Performance Shaders) for Apple Silicon GPUs
    2. CUDA for NVIDIA GPUs
    3. CPU as fallback

    Returns:
        torch.device: The optimal device for inference
    """
    global _DEVICE_CACHE  # pylint: disable=global-statement

    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    if not ENABLE_GPU:
        logger.info("GPU disabled by configuration, using CPU")
        _DEVICE_CACHE = "cpu"
        return _DEVICE_CACHE

    try:
        import torch  # pylint: disable=import-outside-toplevel

        # Check for MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("Using MPS (Apple Silicon GPU) for acceleration")
                _DEVICE_CACHE = "mps"
                return _DEVICE_CACHE

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            logger.info("Using CUDA (NVIDIA GPU) for acceleration")
            _DEVICE_CACHE = "cuda"
            return _DEVICE_CACHE

        # Fallback to CPU
        logger.info("No GPU available, using CPU")
        _DEVICE_CACHE = "cpu"
        return _DEVICE_CACHE

    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        _DEVICE_CACHE = "cpu"
        return _DEVICE_CACHE


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available and enabled.

    Returns:
        bool: True if GPU is available and enabled
    """
    device = get_device()
    return device != "cpu"


def should_use_fp16() -> bool:
    """
    Determine if FP16 precision should be used.

    FP16 is faster on GPUs but may cause issues on CPU.

    Returns:
        bool: True if FP16 should be used
    """
    device = get_device()
    return device in ("mps", "cuda")


def is_core_ml_available() -> bool:
    """
    Check if Core ML is available for Apple Silicon acceleration.

    Returns:
        bool: True if Core ML can be used
    """
    global _CORE_ML_AVAILABLE  # pylint: disable=global-statement

    if _CORE_ML_AVAILABLE is not None:
        return _CORE_ML_AVAILABLE

    if not USE_CORE_ML:
        _CORE_ML_AVAILABLE = False
        return False

    if get_device() != "mps":
        _CORE_ML_AVAILABLE = False
        return False

    try:
        import importlib
        if importlib.util.find_spec(
                "coremltools") is None:  # type: ignore[attr-defined]
            _CORE_ML_AVAILABLE = False
            logger.info("Core ML not available (install coremltools)")
            return False
        _CORE_ML_AVAILABLE = True
        logger.info("Core ML available for Neural Engine acceleration")
        return True
    except ImportError:
        _CORE_ML_AVAILABLE = False
        logger.info("Core ML not available (install coremltools)")
        return False


def ensure_coreml_cache_dir():
    """
    Ensure the Core ML cache directory exists.

    Returns:
        str: Path to the cache directory
    """
    os.makedirs(CORE_ML_CACHE_DIR, exist_ok=True)
    return CORE_ML_CACHE_DIR


def _warmup_mps_kernels(model):
    """
    Warm up MPS Metal kernels with a dummy inference.

    Args:
        model: The model to warm up
    """
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        dummy_audio = np.random.randn(16000).astype(np.float32)
        logger.info("Warming up MPS Metal kernels...")
        _ = model.transcribe(
            dummy_audio,
            language="zh",
            verbose=False,
            condition_on_previous_text=False,
            fp16=False  # Use FP32 for warmup
        )
        logger.info("MPS Metal kernels warmed up")
    except (RuntimeError, ValueError) as warmup_error:
        logger.warning(
            "Failed to warm up MPS kernels: %s", str(warmup_error))


def get_cached_whisper_model(model_size: str):
    """
    Get cached Whisper model or load if not cached.

    This significantly improves performance by avoiding model reloading
    for each transcription. Models are cached globally across the session.
    Loads model on GPU if available.
    """
    cache_key = f"whisper_{model_size}"
    if cache_key not in _MODEL_CACHE:
        try:
            import whisper  # pylint: disable=import-outside-toplevel

            logger.info("Loading Whisper model: %s", model_size)
            model = whisper.load_model(model_size)

            # Move model to GPU if available
            device = get_device()
            if device != "cpu":
                try:
                    # Move model to device
                    model = model.to(device)
                    logger.info("Whisper model moved to %s", device)

                    # Warm up with dummy inference to optimize Metal kernels
                    if device == "mps":
                        _warmup_mps_kernels(model)

                except (RuntimeError, ValueError) as gpu_error:
                    logger.warning(
                        "Failed to move model to GPU: %s. Using CPU.", str(gpu_error))
                    model = model.cpu()

            _MODEL_CACHE[cache_key] = model
            logger.info("Whisper model loaded and cached")

        except ImportError as e:
            raise ImportError(f"Whisper module not available: {e}") from e
        except (RuntimeError, ValueError, OSError) as e:
            raise RuntimeError(
                f"Failed to load Whisper model {model_size}: {e}") from e

    return _MODEL_CACHE[cache_key]


def get_cached_diarization_pipeline():
    """
    Get cached diarization pipeline or load if not cached.

    This significantly improves performance by avoiding pipeline reloading
    for each audio file. Pipeline is cached globally across the session.
    Moves pipeline to GPU if available.
    """
    cache_key = 'diarization_pipeline'
    if cache_key not in _MODEL_CACHE:
        try:
            from pyannote.audio import Pipeline  # pylint: disable=import-outside-toplevel

            logger.info("Loading pyannote speaker diarization pipeline...")

            # Get HuggingFace token from environment
            hf_token = os.environ.get(
                'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

            if hf_token:
                pipeline = Pipeline.from_pretrained(
                    DIARIZATION_MODEL,
                    use_auth_token=hf_token
                )
            else:
                pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)

            # Move pipeline to GPU if available
            device = get_device()
            if device != "cpu":
                try:
                    import torch  # pylint: disable=import-outside-toplevel
                    # Try to move to GPU
                    pipeline = pipeline.to(torch.device(device))
                    logger.info("Diarization pipeline moved to %s", device)
                except (RuntimeError, ValueError) as gpu_error:
                    logger.warning(
                        "Failed to move pipeline to GPU: %s. Using CPU.", str(gpu_error))

            _MODEL_CACHE[cache_key] = pipeline
            logger.info("Diarization pipeline loaded and cached")

        except ImportError as e:
            raise ImportError(f"pyannote.audio not available: {e}") from e
        except (RuntimeError, ValueError, OSError) as e:
            raise RuntimeError(
                f"Failed to load diarization pipeline: {e}") from e

    return _MODEL_CACHE[cache_key]


def cleanup_model_cache():
    """
    Clean up model cache to free memory.

    Useful for long-running processes or when memory is limited.
    """
    for key in list(_MODEL_CACHE.keys()):
        if hasattr(_MODEL_CACHE[key], 'cpu'):
            _MODEL_CACHE[key] = _MODEL_CACHE[key].cpu()

    import gc  # pylint: disable=import-outside-toplevel
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
    import multiprocessing  # pylint: disable=import-outside-toplevel

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
