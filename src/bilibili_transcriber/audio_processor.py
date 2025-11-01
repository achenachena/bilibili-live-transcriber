"""
Audio processing module for extracting and converting audio from videos.
Uses functional programming approach.
"""
import logging
import os
import subprocess
from typing import List, Optional

from .config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_RESET_TIMESTAMPS,
    DEFAULT_SEGMENT_DURATION,
    SUPPORTED_AUDIO_EXTENSIONS
)

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_name: Optional[str] = None,
                  audio_dir: str = "audio", sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
    """
    Extract audio from a video file.
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

    # Use ffmpeg to extract audio with hardware acceleration on macOS
    command = [
        'ffmpeg',
        '-hwaccel', 'auto',  # Use hardware acceleration if available
        '-i', video_path,
        '-ar', str(sample_rate),
        '-ac', str(AUDIO_CHANNELS),  # Mono
        '-acodec', AUDIO_CODEC,  # PCM 16-bit
        '-threads', '0',  # Use all available threads
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


def extract_and_split_audio(video_path: str, segment_duration: int = DEFAULT_SEGMENT_DURATION,
                            audio_dir: str = "audio", sample_rate: int = AUDIO_SAMPLE_RATE,
                            max_duration: Optional[int] = None) -> List[str]:
    """
    Extract audio and optionally split it into segments if the file is too long.

    Automatically splits videos longer than max_duration into smaller segments.
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)

    # Check video duration
    try:
        duration = get_video_duration(video_path)
        logger.info("Video duration: %.2f seconds", duration)

        # If video is long and max_duration is set, split it
        if max_duration is not None and duration > max_duration:
            logger.info(
                "Video exceeds %ds, splitting into segments...",
                max_duration)
            # First extract audio, then split
            temp_audio = extract_audio(
                video_path, None, audio_dir, sample_rate)
            return split_audio(temp_audio, segment_duration, audio_dir)

    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Could not determine video duration: %s", str(e))

    # Extract single audio file
    audio_path = extract_audio(video_path, None, audio_dir, sample_rate)
    return [audio_path]


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrapper=1:nokey=1',
        video_path
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error("Failed to get video duration: %s", str(e))
        raise


def check_audio_file(audio_path: str) -> str:
    """
    Check if an audio file exists and return its path.
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get file extension
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning("Unusual audio format: %s", ext)

    return audio_path


def split_audio(audio_path: str, segment_duration: int = DEFAULT_SEGMENT_DURATION,
                audio_dir: str = "audio") -> List[str]:
    """
    Split a long audio file into smaller segments.
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)

    logger.info(
        "Splitting audio into %ds segments: %s",
        segment_duration,
        audio_path)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Use ffmpeg to split audio
    output_pattern = os.path.join(audio_dir, f"{base_name}_segment_%03d.wav")

    command = [
        'ffmpeg',
        '-hwaccel', 'auto',  # Use hardware acceleration if available
        '-i', audio_path,
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-ar', str(AUDIO_SAMPLE_RATE),  # 16kHz
        '-ac', str(AUDIO_CHANNELS),  # Mono
        '-acodec', AUDIO_CODEC,  # PCM 16-bit
        '-reset_timestamps', str(AUDIO_RESET_TIMESTAMPS),
        '-threads', '0',  # Use all available threads
        '-y',  # Overwrite output file
        output_pattern
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("Audio split successfully")

        # Find all generated segments
        segments = []
        counter = 0
        while True:
            segment_path = os.path.join(
                audio_dir, f"{base_name}_segment_{
                    counter:03d}.wav")
            if os.path.exists(segment_path):
                segments.append(segment_path)
                counter += 1
            else:
                break

        logger.info("Generated %d audio segments", len(segments))
        return segments

    except subprocess.CalledProcessError as e:
        logger.error("Audio split failed: %s", e.stderr.decode()
                     if e.stderr else str(e), exc_info=True)
        raise
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        raise
