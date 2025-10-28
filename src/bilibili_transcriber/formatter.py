"""
Output formatting module for generating text files.
Uses functional programming approach.
"""
import logging
import os
from datetime import timedelta
from typing import List, Tuple

logger = logging.getLogger(__name__)


def convert_to_simplified(text: str) -> str:
    """
    Convert traditional Chinese to simplified Chinese.

    Args:
        text: Text in traditional Chinese

    Returns:
        Text in simplified Chinese
    """
    try:
        import opencc  # pylint: disable=import-outside-toplevel
        converter = opencc.OpenCC('t2s')  # Traditional to Simplified
        return converter.convert(text)
    except ImportError:
        # If opencc is not available, return original text
        logger.warning("opencc not available, returning original text")
        return text


def format_time(seconds: float) -> str:
    """
    Format time in seconds to [HH:MM:SS] format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    td = timedelta(seconds=float(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_segments(segments: List[Tuple[float, float, str, str]]) -> str:
    """
    Format merged segments into readable text.

    Args:
        segments: List of (start, end, speaker, text) tuples

    Returns:
        Formatted text string
    """
    lines = []

    for start, _end, speaker, text in segments:
        time_str = format_time(start)
        lines.append(f"[{time_str}] {speaker}: {text}")

    return "\n".join(lines)


def save_transcription(segments: List[Tuple[float, float, str, str]],
                       output_filename: str, output_dir: str = "output") -> str:
    """
    Save formatted segments to a file.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        segments: List of (start, end, speaker, text) tuples
        output_filename: Output filename
        output_dir: Output directory

    Returns:
        Path to saved file

    Raises:
        OSError: If file cannot be written
    """
    if not output_dir:
        raise ValueError("Output directory cannot be empty")

    if not output_filename:
        raise ValueError("Output filename cannot be empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    formatted_text = format_segments(segments)

    # Convert to simplified Chinese
    formatted_text = convert_to_simplified(formatted_text)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        logger.info("Output saved to: %s", output_path)
        return output_path

    except OSError as e:
        logger.error("Failed to save output: %s", str(e), exc_info=True)
        raise


def generate_filename(source_name: str, extension: str = ".txt") -> str:
    """
    Generate an output filename based on source.

    Args:
        source_name: Source file or URL name
        extension: Output file extension

    Returns:
        Output filename
    """
    if not source_name:
        return f"transcript{extension}"

    if os.path.exists(source_name):
        base_name = os.path.splitext(os.path.basename(source_name))[0]
    else:
        base_name = source_name.replace('/', '_').replace(':', '_')

    return f"{base_name}_transcript{extension}"
