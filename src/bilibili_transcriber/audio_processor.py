"""
Audio processing module for extracting and converting audio from videos.
Uses functional programming approach.
"""
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_name: Optional[str] = None,
                  audio_dir: str = "audio", sample_rate: int = 16000) -> str:
    """
    Extract audio from a video file.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        video_path: Path to video file
        output_name: Optional output filename
        audio_dir: Directory to save audio files
        sample_rate: Sample rate for audio conversion

    Returns:
        Path to the extracted audio file

    Raises:
        FileNotFoundError: If video file not found
        subprocess.CalledProcessError: If ffmpeg fails
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)

    logger.info("Extracting audio from: %s", video_path)

    # Generate output filename
    if output_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}.wav"

    output_path = os.path.join(audio_dir, output_name)

    # Use ffmpeg to extract audio
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ar', str(sample_rate),
        '-ac', '1',  # Mono
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-y',  # Overwrite output file
        output_path
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("Audio extracted to: %s", output_path)
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error("Audio extraction failed: %s", e.stderr.decode()
                     if e.stderr else str(e), exc_info=True)
        raise
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        raise


def check_audio_file(audio_path: str) -> str:
    """
    Check if an audio file exists and return its path.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        audio_path: Path to audio file

    Returns:
        Validated audio file path

    Raises:
        FileNotFoundError: If audio file not found
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get file extension
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        logger.warning("Unusual audio format: %s", ext)

    return audio_path
