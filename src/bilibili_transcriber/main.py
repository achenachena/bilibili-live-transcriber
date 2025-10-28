"""
Main CLI interface for Bilibili Live Transcriber.
Uses functional programming approach.
"""
import logging
import os
import sys
from typing import Optional

import click

from . import (audio_processor, diarizer, downloader, formatter, merger,
               transcriber)
from .config import (AUDIO_DIR, OUTPUT_DIR, VIDEO_DIR, WHISPER_LANGUAGE,
                     WHISPER_MODEL)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_bilibili_url(url: str, output_name: Optional[str] = None,
                         whisper_model: str = WHISPER_MODEL,
                         whisper_language: str = WHISPER_LANGUAGE) -> str:
    """
    Process a Bilibili video URL through the full pipeline.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        url: Bilibili video URL
        output_name: Optional output filename
        whisper_model: Whisper model size
        whisper_language: Language code

    Returns:
        Path to output file

    Raises:
        ValueError: If url is invalid
        Exception: If any step in pipeline fails
    """
    if not url:
        raise ValueError("URL cannot be empty")

    logger.info("Processing Bilibili URL: %s", url)

    try:
        # Step 1: Download video
        logger.info("Step 1/5: Downloading video...")
        video_path = downloader.download_video(url, VIDEO_DIR)

        # Step 2: Extract audio
        logger.info("Step 2/5: Extracting audio...")
        audio_path = audio_processor.extract_audio(
            video_path, output_name, AUDIO_DIR)

        # Step 3-5: Process audio
        return process_audio_file(audio_path, output_name, whisper_model, whisper_language)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


def process_local_file(file_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> str:
    """
    Process a local video/audio file through the full pipeline.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        file_path: Path to video or audio file
        output_name: Optional output filename
        whisper_model: Whisper model size
        whisper_language: Language code

    Returns:
        Path to output file

    Raises:
        ValueError: If file_path is invalid
        FileNotFoundError: If file not found
        Exception: If any step in pipeline fails
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Processing file: %s", file_path)

    try:
        # Check if it's an audio or video file
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            # It's an audio file
            audio_path = audio_processor.check_audio_file(file_path)
        else:
            # It's a video file, extract audio
            logger.info("Extracting audio from video...")
            audio_path = audio_processor.extract_audio(
                file_path, output_name, AUDIO_DIR)

        # Process audio
        return process_audio_file(audio_path, output_name, whisper_model, whisper_language)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


def process_audio_file(audio_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> str:
    """
    Process audio file with diarization and transcription.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        audio_path: Path to audio file
        output_name: Optional output filename
        whisper_model: Whisper model size
        whisper_language: Language code

    Returns:
        Path to output file
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    # Step 3: Speaker diarization
    logger.info("Step 3/5: Performing speaker diarization...")
    diarization_segments = diarizer.diarize_audio(audio_path)

    # Step 4: Transcription
    logger.info("Step 4/5: Transcribing audio...")
    transcription_segments = transcriber.get_text_with_timestamps(
        audio_path, whisper_model, whisper_language
    )

    # Step 5: Merge and save
    logger.info("Step 5/5: Merging results and formatting output...")
    merged_segments = merger.merge_results(
        diarization_segments, transcription_segments)

    # Generate output filename
    if output_name is None:
        output_name = formatter.generate_filename(audio_path)

    output_path = formatter.save_transcription(
        merged_segments, output_name, OUTPUT_DIR)

    logger.info("Transcription complete!")
    logger.info("Output saved to: %s", output_path)

    return output_path


@click.command()
@click.argument('input_source')
@click.option('--file', 'is_file', is_flag=True, help='Treat input as a file path instead of URL')
@click.option('--output-dir', default=OUTPUT_DIR, help='Output directory for transcriptions')
@click.option('--model', 'whisper_model', default=WHISPER_MODEL,
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--whisper-language', default=WHISPER_LANGUAGE,
              help='Language code for Whisper (e.g., zh, en, ja)')
def main(input_source: str, is_file: bool, output_dir: str,
         whisper_model: str, whisper_language: str) -> None:
    """
    Transcribe Bilibili live recordings with speaker diarization.

    \b
    Examples:
        python main.py https://www.bilibili.com/video/BVxxxxx
        python main.py --file video.mp4
        python main.py https://www.bilibili.com/video/BVxxxxx --model large
    """
    # Update output directory if specified
    if output_dir != OUTPUT_DIR:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Update the config OUTPUT_DIR for this session
        globals()['OUTPUT_DIR'] = output_dir

    try:
        if is_file:
            output_path = process_local_file(input_source, whisper_model=whisper_model,
                                             whisper_language=whisper_language)
        else:
            output_path = process_bilibili_url(input_source, whisper_model=whisper_model,
                                               whisper_language=whisper_language)

        click.echo(f"\n✓ Transcription saved to: {output_path}")

    except (ValueError, FileNotFoundError) as e:
        logger.error("Error: %s", str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        # Catch-all for unexpected errors to prevent crashes
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        click.echo(
            "Error: An unexpected error occurred. Check logs for details.", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
