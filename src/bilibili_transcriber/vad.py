"""
Voice Activity Detection (VAD) module using Silero VAD.
Uses functional programming approach.
"""
import logging
from typing import List, Tuple
import importlib

logger = logging.getLogger(__name__)

# VAD model cache
_VAD_MODEL_CACHE = None
_VAD_UTILS_CACHE = None


def _load_vad_model():
    """Load Silero VAD and return (model, utils)."""
    global _VAD_MODEL_CACHE, _VAD_UTILS_CACHE  # pylint: disable=global-statement

    if _VAD_MODEL_CACHE is not None and _VAD_UTILS_CACHE is not None:
        return _VAD_MODEL_CACHE, _VAD_UTILS_CACHE

    try:
        import torch  # pylint: disable=import-outside-toplevel
        # runtime import to avoid static issues
        silero_mod = importlib.import_module("silero_vad")
        load_silero_vad = getattr(silero_mod, "load_silero_vad")

        # Load Silero VAD model
        device = torch.device("cpu")  # Silero VAD works best on CPU
        vad_model, vad_utils = load_silero_vad(onnx=True)  # Use ONNX for speed

        vad_model = vad_model.to(device)

        _VAD_MODEL_CACHE = vad_model
        _VAD_UTILS_CACHE = vad_utils

        logger.info("Silero VAD model loaded successfully")
        return vad_model, vad_utils

    except ImportError:
        logger.warning(
            "Silero VAD not available. Install with: pip install silero-vad")
        raise ImportError(
            "Silero VAD required. Install with: pip install silero-vad") from None
    except Exception as e:
        logger.error("Failed to load Silero VAD model: %s", str(e))
        raise


def detect_voice_activity(
    audio_path: str,
    sampling_rate: int = 16000,
    min_silence_duration_ms: int = 500,  # pylint: disable=unused-argument
    min_speech_duration_ms: int = 250
) -> List[Tuple[float, float]]:
    """Return a list of (start, end) speech segments detected by VAD."""
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    logger.info("Detecting voice activity in: %s", audio_path)

    try:
        import torch  # pylint: disable=import-outside-toplevel
        import torchaudio  # pylint: disable=import-outside-toplevel

        # Load VAD model
        vad_model, vad_utils = _load_vad_model()

        # Load audio
        waveform, source_sr = torchaudio.load(audio_path)

        # Resample if needed
        if source_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(
                source_sr, sampling_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Get VAD timestamps
        (
            speech_timestamps,
            _speech_probabilities
        ) = vad_utils[0](waveform, vad_model, sampling_rate=sampling_rate)

        # Filter by minimum speech duration
        filtered_timestamps = []
        for timestamp in speech_timestamps:
            start_ms = timestamp['start']
            end_ms = timestamp['end']
            duration_ms = end_ms - start_ms

            if duration_ms >= min_speech_duration_ms:
                filtered_timestamps.append(
                    (start_ms / 1000.0, end_ms / 1000.0))

        logger.info(
            "Detected %d speech segments (total: %d before filtering)",
            len(filtered_timestamps),
            len(speech_timestamps)
        )

        return filtered_timestamps

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Voice activity detection failed: %s",
            str(e),
            exc_info=True)
        # On failure, skip VAD and keep all segments by returning empty list
        logger.warning(
            "VAD failed, skipping VAD and keeping all segments")
        return []


def filter_audio_by_vad(audio_segments: List[Tuple[float, float]],
                        vad_segments: List[Tuple[float, float]]) -> List[Tuple[float, float, str]]:
    """Return only segments overlapping VAD regions."""
    if not vad_segments:
        logger.warning("No VAD segments, returning all audio segments")
        return audio_segments

    filtered = []
    vad_idx = 0

    for start, end, text in audio_segments:
        # Check if this segment overlaps with any VAD segment
        has_overlap = False

        while vad_idx < len(vad_segments):
            vad_start, vad_end = vad_segments[vad_idx]

            # Check overlap
            if start < vad_end and end > vad_start:
                has_overlap = True
                break

            # Move to next VAD segment if current one ends before audio segment
            if vad_end <= start:
                vad_idx += 1
            else:
                break

        if has_overlap:
            filtered.append((start, end, text))
        else:
            logger.debug(
                "Filtering out non-voice segment: %.2f-%.2f",
                start, end)

    logger.info(
        "VAD filtering: %d segments kept out of %d",
        len(filtered),
        len(audio_segments)
    )

    return filtered
