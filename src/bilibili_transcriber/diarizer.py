"""
Speaker diarization module using pyannote.audio.
Uses functional programming approach.
"""
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from .config import DIARIZATION_MODEL, AUDIO_BITS_PER_SAMPLE, get_cached_diarization_pipeline

# Suppress TorchCodec backend warnings
warnings.filterwarnings(
    "ignore",
    message="The 'backend' parameter is not used by TorchCodec AudioDecoder"
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Set soundfile as the default backend for torchaudio to avoid torchcodec
# issues
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# Workaround for torchaudio API changes in torchaudio 2.9.0 with Python 3.14
try:
    import torchaudio
    # Workaround 1: list_audio_backends
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _dummy_list_audio_backends():
            return ["soundfile"]
        torchaudio.list_audio_backends = _dummy_list_audio_backends

    # Workaround 2: AudioMetaData (deprecated in torchaudio 2.9.0)
    if not hasattr(torchaudio, 'AudioMetaData'):
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:  # type: ignore
            """Workaround for deprecated torchaudio.AudioMetaData."""
            num_frames: int
            num_channels: int
            sample_rate: int
            bits_per_sample: int = AUDIO_BITS_PER_SAMPLE

        torchaudio.AudioMetaData = AudioMetaData

    # Workaround 3: info() method (removed in nightly builds)
    if torchaudio is not None and not hasattr(torchaudio, 'info'):
        # type: ignore  # pylint: disable=unused-argument
        def _torchaudio_info(uri, backend=None):
            """Workaround for missing torchaudio.info."""
            import soundfile as sf  # pylint: disable=import-outside-toplevel
            info = sf.info(uri)

            # Use AudioMetaData class already defined above for consistency
            return AudioMetaData(  # type: ignore
                num_frames=info.frames,
                num_channels=info.channels,
                sample_rate=info.samplerate
            )

        torchaudio.info = _torchaudio_info
except ImportError:
    torchaudio = None  # type: ignore

# Import heavy ML libraries at top level to avoid
# "import-outside-toplevel" warnings
try:
    from pyannote.audio import Pipeline  # noqa
except ImportError:
    Pipeline = None  # type: ignore

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, model_name: str = DIARIZATION_MODEL,  # pylint: disable=unused-argument
                  min_speakers: Optional[int] = None,
                  max_speakers: Optional[int] = None) -> List[Tuple[float, float, str]]:
    """Perform speaker diarization and return (start, end, speaker) tuples."""
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    logger.info("Getting cached speaker diarization pipeline...")

    # Import check
    if Pipeline is None:
        raise ImportError("pyannote.audio not available")

    try:
        pipeline = get_cached_diarization_pipeline()
        logger.info("Diarization pipeline retrieved from cache successfully")
    except Exception as e:
        logger.error(
            "Failed to get diarization pipeline: %s",
            str(e),
            exc_info=True)
        raise

    logger.info("Diarizing audio: %s", audio_path)

    try:
        diarization = pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        speaker_count = len(set(s[2] for s in segments))
        logger.info("Found %d different speakers", speaker_count)
        logger.info("Detected %d speaker segments", len(segments))

        return segments

    except Exception as e:
        logger.error("Diarization failed: %s", str(e), exc_info=True)
        raise
