"""
Speech transcription module using OpenAI Whisper.
Uses functional programming approach.
"""
import logging
import sys
import re
from typing import Any, Dict, List, Optional, Tuple

# Note: WHISPER_INITIAL_PROMPTS removed due to repetition issues
from .config import (
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_TEMPERATURE,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_WORD_TIMESTAMPS,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    ENABLE_VAD,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_MIN_SPEECH_DURATION_MS,
    REPETITIVE_SEGMENT_THRESHOLD,
    MIN_SEGMENT_LENGTH,
    MAX_TEXT_PREVIEW_LENGTH,
    get_cached_whisper_model,
    should_use_fp16
)

# Import VAD module conditionally
try:
    from . import vad  # noqa
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

logger = logging.getLogger(__name__)


def _patch_tiktoken_for_python314():
    """
    Patch tiktoken tokenizer to handle OverflowError on Python 3.14.
    This is a workaround for tiktoken compatibility issues with Python 3.14.
    """
    # Only patch on Python 3.14+
    if sys.version_info < (3, 14):
        return

    try:
        import tiktoken
        import tiktoken.core

        # Store original decode method
        original_decode = tiktoken.core.Encoding.decode

        def patched_decode(self, token_ids, **kwargs):
            """Patched decode that handles OverflowError."""
            try:
                return original_decode(self, token_ids, **kwargs)
            except OverflowError as e:
                # Filter out invalid token IDs and retry
                if "out of range integral type conversion" in str(e):
                    logger.warning(
                        "tiktoken OverflowError detected, filtering invalid tokens")
                    # Convert to list if needed and filter
                    token_list = list(token_ids) if not isinstance(
                        token_ids, list) else token_ids
                    # Filter out values that are out of range for uint16
                    # tiktoken uses uint16 for token IDs
                    max_valid_token = 65535  # 2^16 - 1
                    filtered_tokens = [
                        t for t in token_list
                        if isinstance(t, int) and 0 <= t <= max_valid_token
                    ]
                    if filtered_tokens:
                        logger.debug(
                            "Filtered %d invalid tokens, retrying decode",
                            len(token_list) - len(filtered_tokens))
                        return original_decode(self, filtered_tokens, **kwargs)
                    # If all tokens filtered, return empty string
                    logger.warning(
                        "All tokens filtered out, returning empty string")
                    return ""
                raise

        # Apply patch
        tiktoken.core.Encoding.decode = patched_decode
        logger.debug("Applied tiktoken Python 3.14 compatibility patch")
    except ImportError:
        # tiktoken not available, skip patching
        pass
    except Exception as e:  # pylint: disable=broad-except
        # Catch all exceptions during patching as we don't know
        # what might go wrong when monkey-patching third-party code
        logger.warning(
            "Failed to patch tiktoken for Python 3.14: %s", str(e))


# Apply patch on module import
_patch_tiktoken_for_python314()


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

    # Whisper import is handled in get_cached_whisper_model

    logger.info("Getting cached Whisper model: %s", model_size)

    try:
        model = get_cached_whisper_model(model_size)
        logger.info("Whisper model retrieved from cache successfully")
    except Exception as e:
        logger.error("Failed to get Whisper model: %s", str(e), exc_info=True)
        raise

    logger.info("Transcribing audio: %s", audio_path)

    try:
        # Enhanced anti-hallucination parameters based on community solutions
        # Use FP16 on GPU for better performance
        use_fp16 = should_use_fp16()

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
            fp16=use_fp16,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            suppress_tokens=[-1]  # use library defaults
        )

        logger.info("Transcription complete: %d segments",
                    len(result['segments']))

        # Quality check: if most segments look like gibberish, retry with safer
        # settings
        if _transcription_looks_like_gibberish(result):
            logger.warning(
                "Transcription appears to be gibberish; retrying with safer settings (fp16=False, beam search, temperature=0.2)")
            try:
                result_retry = model.transcribe(
                    audio_path,
                    language=language or "zh",
                    verbose=verbose,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    temperature=0.2,
                    compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                    logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
                    no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                    fp16=False,
                    beam_size=max(WHISPER_BEAM_SIZE, 5),
                    best_of=max(WHISPER_BEST_OF, 3),
                    suppress_tokens=[-1]
                )
                if not _transcription_looks_like_gibberish(result_retry):
                    logger.info("Retry improved quality; using retried result")
                    return result_retry
                logger.warning(
                    "Retry still looks like gibberish; keeping original result")
            except Exception as retry_e:  # pylint: disable=broad-except
                logger.warning("Safe retry failed: %s", str(retry_e))
                # As a last resort, try CPU decoding to avoid MPS instability
                try:
                    logger.warning(
                        "Retrying on CPU due to potential MPS instability...")
                    try:
                        original_device = next(
                            # type: ignore[attr-defined]
                            model.parameters()).device
                    except Exception:  # pylint: disable=broad-except
                        original_device = None
                    model.to("cpu")
                    result_cpu = model.transcribe(
                        audio_path,
                        language=language or "zh",
                        verbose=verbose,
                        word_timestamps=False,
                        condition_on_previous_text=False,
                        temperature=0.2,
                        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                        logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
                        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                        fp16=False,
                        beam_size=max(WHISPER_BEAM_SIZE, 5),
                        best_of=max(WHISPER_BEST_OF, 3),
                        suppress_tokens=[-1]
                    )
                    if original_device is not None and str(
                            original_device).startswith("mps"):
                        model.to("mps")
                    if not _transcription_looks_like_gibberish(result_cpu):
                        logger.info(
                            "CPU retry improved quality; using CPU result")
                        return result_cpu
                    logger.warning(
                        "CPU retry still looks like gibberish; keeping original result")
                except Exception as cpu_retry_e:  # pylint: disable=broad-except
                    logger.warning("CPU retry failed: %s", str(cpu_retry_e))

        return result

    except OverflowError as e:
        # Handle tiktoken OverflowError on Python 3.14
        error_msg = str(e)
        if "out of range integral type conversion" in error_msg:
            logger.error(
                "Transcription failed due to tiktoken Python 3.14 compatibility "
                "issue. This is a known issue with tiktoken and Python 3.14. "
                "Consider using Python 3.13 or wait for tiktoken update. "
                "Error: %s", error_msg)
            logger.error(
                "The tiktoken patch may not have been sufficient. "
                "Trying alternative approach...")
            # Try with different parameters that might avoid the issue
            try:
                logger.info("Retrying with modified parameters...")
                result = model.transcribe(
                    audio_path,
                    language=language,
                    verbose=verbose,
                    word_timestamps=False,  # Disable word timestamps
                    condition_on_previous_text=False,
                    temperature=0.0,
                    fp16=use_fp16,
                    beam_size=1,  # Use greedy decoding
                    best_of=1
                )
                logger.info("Transcription complete with fallback parameters: "
                            "%d segments", len(result['segments']))
                return result
            except Exception as retry_error:
                logger.error("Fallback transcription also failed: %s",
                             str(retry_error), exc_info=True)
                raise RuntimeError(
                    "Transcription failed due to tiktoken Python 3.14 "
                    "compatibility issue. Please use Python 3.13 or update "
                    "tiktoken.") from e
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", str(e), exc_info=True)
        raise


def get_text_with_timestamps(audio_path: str, model_size: str = "base",
                             language: Optional[str] = "zh") -> List[Tuple[float, float, str]]:
    """
    Get transcription with timestamps formatted for output.

    Includes automatic filtering for repetitive segments.
    Optionally applies VAD pre-filtering if enabled.
    """
    result = transcribe_audio(audio_path, model_size, language)

    segments = []
    for seg in result['segments']:
        segments.append((
            seg['start'],
            seg['end'],
            seg['text'].strip()
        ))

    # Apply VAD post-filtering if enabled and available
    if ENABLE_VAD and VAD_AVAILABLE:
        try:
            logger.info("Applying VAD post-filtering...")
            vad_segments = vad.detect_voice_activity(
                audio_path,
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS
            )
            segments = vad.filter_audio_by_vad(segments, vad_segments)
        except (ImportError, OSError, ValueError, RuntimeError) as vad_error:
            logger.warning(
                "VAD filtering failed: %s. Continuing without VAD.",
                str(vad_error))

    # Filter out repetitive segments
    filtered_segments = _filter_repetitive_segments(segments)

    return filtered_segments


def _filter_repetitive_segments(
        segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    Filter out repetitive segments that are likely hallucinations.

    Addresses known Whisper issue (GitHub #2015, #1962) where model generates
    repetitive text even when audio content is not repetitive.

    Improved logic: Only filter if text appears many times AND represents
    a significant portion of total segments, indicating likely hallucination.
    """
    if not segments:
        return segments

    filtered = []
    repetitive_threshold = REPETITIVE_SEGMENT_THRESHOLD

    # Count occurrences of each text
    text_counts = {}
    for _, _, text in segments:
        text_counts[text] = text_counts.get(text, 0) + 1

    # Calculate stats for better filtering
    total_segments = len(segments)
    # Only consider it repetitive if it appears many times AND represents
    # a significant portion (>10%) of all segments
    min_repetitive_count = max(repetitive_threshold,
                               int(total_segments * 0.10))

    # Track what we're filtering
    filtered_stats = {
        'repetitive': 0,
        'too_short': 0,
        'punctuation_only': 0,
        'gibberish': 0
    }

    # Filter segments
    for start, end, text in segments:
        # Only skip if text is empty (avoid over-filtering short phrases)
        if not text.strip():
            filtered_stats['too_short'] += 1
            continue

        # Skip if text contains only punctuation or numbers
        if text.strip().replace('，', '').replace('。', '').replace(
                '！', '').replace('？', '').replace(' ', '').isdigit():
            filtered_stats['punctuation_only'] += 1
            continue

        # Skip if text looks like gibberish (mostly punctuation/symbols)
        if _looks_like_gibberish(text):
            filtered_stats['gibberish'] += 1
            continue

        # Only filter as repetitive if it appears many times AND is a significant
        # portion of segments (indicating hallucination, not legitimate
        # repetition)
        count = text_counts[text]
        if count >= min_repetitive_count:
            preview = (text[:MAX_TEXT_PREVIEW_LENGTH] + "..."
                       if len(text) > MAX_TEXT_PREVIEW_LENGTH else text)
            logger.debug("Filtering repetitive segment (appears %d times): %s",
                         count, preview)
            filtered_stats['repetitive'] += 1
            continue

        filtered.append((start, end, text))

    # Enhanced logging
    logger.info(
        "Filtering summary: %d repetitive, %d too short, %d punctuation only, %d gibberish. "
        "Kept %d segments out of %d total",
        filtered_stats['repetitive'],
        filtered_stats['too_short'],
        filtered_stats['punctuation_only'],
        filtered_stats['gibberish'],
        len(filtered),
        total_segments)

    # Safety check: if we filtered everything, something is wrong
    if len(filtered) == 0 and total_segments > 0:
        logger.warning(
            "All segments were filtered! This may indicate overly aggressive "
            "filtering. Keeping original segments as fallback.")
        # Prefer segments with meaningful text (letters/CJK), otherwise fall
        # back to first few
        meaningful: List[Tuple[float, float, str]] = []
        for start, end, text in segments:
            if (len(text.strip()) >= MIN_SEGMENT_LENGTH and
                    _has_meaningful_text(text)):
                meaningful.append((start, end, text))

        if meaningful:
            # Keep up to 200 meaningful segments to avoid empty outputs
            filtered.extend(meaningful[:200])
        else:
            # As a last resort, keep the first 50 non-empty segments regardless
            # of content
            fallback = []
            for start, end, text in segments:
                if text.strip():
                    fallback.append((start, end, text))
                if len(fallback) >= 50:
                    break
            filtered.extend(fallback)

    return filtered


def _looks_like_gibberish(text: str) -> bool:
    """Return True if text is mostly punctuation/symbols."""
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    # Consider common punctuation and symbols likely to appear in
    # hallucinations
    punctuation_symbols = set(
        '!,.:;?%~`^|/\\-_—"\'“”‘’·…<>《》()[]{}#@&*+=，。！？、；：（）【】％')

    total = len(stripped)
    punct_count = sum(1 for ch in stripped if ch in punctuation_symbols)

    # If more than 60% are punctuation/symbols, treat as gibberish
    if total > 0 and (punct_count / total) >= 0.6:
        return True

    # If there is a very long run of the same punctuation, also treat as
    # gibberish
    max_run = 1
    current_run = 1
    for i in range(1, len(stripped)):
        if stripped[i] == stripped[i -
                                   1] and stripped[i] in punctuation_symbols:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    if max_run >= 8:
        return True

    return False


def _has_meaningful_text(text: str) -> bool:
    """Return True if text contains letters or CJK characters."""
    if not text:
        return False
    # Fast path: any alphabetic unicode letter
    if any(ch.isalpha() for ch in text):
        return True
    # Check CJK Unified Ideographs range explicitly
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _transcription_looks_like_gibberish(result: Dict[str, Any]) -> bool:
    """Return True if most segments are gibberish or text is punctuation-heavy."""
    try:
        segments = result.get('segments') or []
        if not segments:
            return True

        texts = [str(seg.get('text', '')).strip() for seg in segments]
        if not texts:
            return True

        gibberish_flags = [_looks_like_gibberish(t) for t in texts]
        gibberish_ratio = sum(
            1 for f in gibberish_flags if f) / float(len(texts))

        if gibberish_ratio >= 0.5:
            return True

        # Also evaluate concatenated content
        combined = " ".join(texts)
        punctuation_symbols = set(
            '!,.:;?%~`^|/\\-_—"\'“”‘’·…<>《》()[]{}#@&*+=，。！？、；：（）【】％')
        total = len(combined)
        if total == 0:
            return True
        punct_count = sum(1 for ch in combined if ch in punctuation_symbols)
        if (punct_count / total) >= 0.6:
            return True

        return False
    except Exception:  # pylint: disable=broad-except
        # On any failure, assume not gibberish to avoid over-triggering retries
        return False
