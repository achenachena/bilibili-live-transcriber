"""
Speech transcription module using OpenAI Whisper.
Uses functional programming approach.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import WHISPER_INITIAL_PROMPTS

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, model_size: str = "base",
                     language: Optional[str] = "zh", verbose: bool = False) -> Dict[str, Any]:
    """
    Transcribe audio file to text with timestamps.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        audio_path: Path to audio file
        model_size: Size of Whisper model (tiny, base, small, medium, large)
        language: Language code (e.g., 'zh', 'en', 'ja') or None for auto-detect
        verbose: Whether to print progress

    Returns:
        Dictionary containing 'segments' list with transcription and timestamps

    Raises:
        ImportError: If whisper module not available
        Exception: If transcription fails
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    try:
        import whisper  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        logger.error("Whisper module not available: %s", str(e))
        raise

    logger.info("Loading Whisper model: %s", model_size)

    try:
        model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error("Failed to load Whisper model: %s", str(e), exc_info=True)
        raise

    logger.info("Transcribing audio: %s", audio_path)

    try:
        # Use minimal initial prompt to guide punctuation without hallucinations
        initial_prompt = WHISPER_INITIAL_PROMPTS.get(language)

        transcribe_params = {
            "language": language,
            "verbose": verbose,
            "word_timestamps": False  # Use segment-level timestamps
        }

        # Only add prompt if we have one for this language
        if initial_prompt:
            transcribe_params["initial_prompt"] = initial_prompt

        result = model.transcribe(audio_path, **transcribe_params)

        logger.info("Transcription complete: %d segments",
                    len(result['segments']))
        return result

    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        raise


def get_text_with_timestamps(audio_path: str, model_size: str = "base",
                             language: Optional[str] = "zh") -> List[Tuple[float, float, str]]:
    """
    Get transcription with timestamps formatted for output.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Language code or None

    Returns:
        List of tuples (start_time, end_time, text)
    """
    result = transcribe_audio(audio_path, model_size, language)

    segments = []
    for seg in result['segments']:
        segments.append((
            seg['start'],
            seg['end'],
            seg['text'].strip()
        ))

    return segments
