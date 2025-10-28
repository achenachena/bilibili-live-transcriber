"""
Speech transcription module using OpenAI Whisper.
Uses functional programming approach.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

# Note: WHISPER_INITIAL_PROMPTS removed due to repetition issues
from .config import (
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_TEMPERATURE,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_FP16,
    WHISPER_WORD_TIMESTAMPS,
    REPETITIVE_SEGMENT_THRESHOLD,
    MIN_SEGMENT_LENGTH,
    MAX_TEXT_PREVIEW_LENGTH
)

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, model_size: str = "base",
                     language: Optional[str] = "zh", verbose: bool = False) -> Dict[str, Any]:
    """
    Transcribe audio file to text with timestamps.

    Uses anti-hallucination parameters to prevent repetitive output issues.
    Addresses known Whisper hallucination problem (GitHub #2015, #1962).
    Parameters are configurable in config.py.
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
        # Enhanced anti-hallucination parameters based on community solutions
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=verbose,
            word_timestamps=WHISPER_WORD_TIMESTAMPS,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            fp16=WHISPER_FP16
        )

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

    Includes automatic filtering for repetitive segments.
    """
    result = transcribe_audio(audio_path, model_size, language)

    segments = []
    for seg in result['segments']:
        segments.append((
            seg['start'],
            seg['end'],
            seg['text'].strip()
        ))

    # Filter out repetitive segments
    filtered_segments = _filter_repetitive_segments(segments)

    return filtered_segments


def _filter_repetitive_segments(
        segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    Filter out repetitive segments that are likely hallucinations.

    Addresses known Whisper issue (GitHub #2015, #1962) where model generates
    repetitive text even when audio content is not repetitive.
    """
    if not segments:
        return segments

    filtered = []
    repetitive_threshold = REPETITIVE_SEGMENT_THRESHOLD

    # Count occurrences of each text
    text_counts = {}
    for _, _, text in segments:
        text_counts[text] = text_counts.get(text, 0) + 1

    # Filter segments
    for start, end, text in segments:
        # Skip if text is too repetitive
        if text_counts[text] >= repetitive_threshold:
            logger.debug("Filtering repetitive segment: %s",
                         text[:MAX_TEXT_PREVIEW_LENGTH] + "..." if len(text) > MAX_TEXT_PREVIEW_LENGTH else text)
            continue

        # Skip if text is too short (likely noise)
        if len(text.strip()) < MIN_SEGMENT_LENGTH:
            continue

        # Skip if text contains only punctuation or numbers
        if text.strip().replace('，', '').replace('。', '').replace(
                '！', '').replace('？', '').replace(' ', '').isdigit():
            continue

        filtered.append((start, end, text))

    logger.info("Filtered %d repetitive segments, kept %d segments",
                len(segments) - len(filtered), len(filtered))

    return filtered
