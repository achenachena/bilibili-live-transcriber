"""
Speaker diarization module using pyannote.audio.
Uses functional programming approach.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Set soundfile as the default backend for torchaudio to avoid torchcodec issues
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
            """Workaround for deprecated AudioMetaData in torchaudio 2.9.0"""
            num_frames: int
            num_channels: int
            sample_rate: int
            bits_per_sample: int = 16

        torchaudio.AudioMetaData = AudioMetaData

    # Workaround 3: info() method (removed in nightly builds)
    if torchaudio is not None and not hasattr(torchaudio, 'info'):
        def _torchaudio_info(uri, backend=None):  # type: ignore  # pylint: disable=unused-argument
            """Workaround for missing torchaudio.info in nightly builds"""
            import soundfile as sf
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

# Import heavy ML libraries at top level to avoid "import-outside-toplevel" warnings
try:
    from pyannote.audio import Pipeline  # noqa
    import torch  # noqa
except ImportError:
    Pipeline = None  # type: ignore
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, model_name: str = "pyannote/speaker-diarization-3.1",
                  min_speakers: Optional[int] = None,
                  max_speakers: Optional[int] = None) -> List[Tuple[float, float, str]]:
    """
    Perform speaker diarization on an audio file.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        audio_path: Path to audio file
        model_name: Diarization model name
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)

    Returns:
        List of tuples (start_time, end_time, speaker_label)

    Raises:
        ImportError: If pyannote.audio not available
        Exception: If diarization fails
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    logger.info("Initializing speaker diarization pipeline...")

    # Import check
    if Pipeline is None:
        raise ImportError("pyannote.audio not available")

    pipeline = _initialize_pipeline(model_name)

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


def _initialize_pipeline(model_name: str):  # type: ignore
    """
    Initialize the diarization pipeline.

    Internal helper function.

    Args:
        model_name: Model name to use

    Returns:
        Initialized pipeline
    """
    # Check dependencies
    if torch is None:
        logger.error("torch not available")
        raise ImportError("torch not available")
    if Pipeline is None:
        logger.error("pyannote.audio not available")
        raise ImportError("pyannote.audio not available")

    try:
        # Get HuggingFace token from environment
        hf_token = os.environ.get(
            'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

        if hf_token:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)  # pylint: disable=unexpected-keyword-arg
        else:
            try:
                pipeline = Pipeline.from_pretrained(model_name)
            except Exception:
                logger.error(
                    "Authentication required. Please set HF_TOKEN environment variable.")
                logger.error(
                    "Visit https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    "to accept the model license.")
                raise

        # Move to GPU if available
        if torch.cuda.is_available():
            logger.info("Using GPU for speaker diarization")
            pipeline = pipeline.to(torch.device("cuda"))

        logger.info("Diarization pipeline initialized")
        return pipeline

    except Exception as e:
        logger.error("Failed to initialize pipeline: %s", str(e), exc_info=True)
        raise
