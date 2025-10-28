"""
Main CLI interface for Bilibili Live Transcriber.
Uses functional programming approach.
"""
import logging
import os
import sys
from typing import List, Optional

import click

from . import (audio_processor, diarizer, downloader, formatter, merger,
               transcriber)
from .config import (AUDIO_DIR, OUTPUT_DIR, VIDEO_DIR, WHISPER_LANGUAGE,
                     WHISPER_MODEL, PIPELINE_STEPS, DEFAULT_SPLIT_THRESHOLD,
                     DEFAULT_SPLIT_THRESHOLD_CLI, DEFAULT_SEGMENT_DURATION,
                     SUPPORTED_AUDIO_EXTENSIONS)

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
    """
    if not url:
        raise ValueError("URL cannot be empty")

    logger.info("Processing Bilibili URL: %s", url)

    video_path = None
    audio_path = None

    try:
        # Step 1: Download video
        logger.info(
            "Step %d/5: Downloading video...",
            PIPELINE_STEPS["DOWNLOAD"])
        video_path = downloader.download_video(url, VIDEO_DIR)

        # Step 2: Extract audio
        logger.info(
            "Step %d/5: Extracting audio...",
            PIPELINE_STEPS["EXTRACT"])
        audio_path = audio_processor.extract_audio(
            video_path, output_name, AUDIO_DIR)

        # Step 3-5: Process audio
        output_path = process_audio_file(
            audio_path, output_name, whisper_model, whisper_language)

        # Clean up temporary files
        if video_path and os.path.exists(video_path):
            logger.info("Deleting temporary video file: %s", video_path)
            os.remove(video_path)
            logger.debug("Video file deleted successfully")

        if audio_path and os.path.exists(audio_path):
            logger.info("Deleting temporary audio file: %s", audio_path)
            os.remove(audio_path)
            logger.debug("Audio file deleted successfully")

        return output_path

    except Exception as e:
        # Clean up temporary files even if processing fails
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.debug("Cleaned up video file after error")
            except Exception:  # pylint: disable=broad-except
                pass
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug("Cleaned up audio file after error")
            except Exception:  # pylint: disable=broad-except
                pass

        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


def process_local_file(file_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE,
                       split_duration: Optional[int] = None) -> str:
    """
    Process a local video/audio file through the full pipeline.

    Automatically splits videos longer than split_duration into segments.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Processing file: %s", file_path)

    # Set default split threshold (2 hours = 7200 seconds)
    max_duration = split_duration if split_duration is not None else DEFAULT_SPLIT_THRESHOLD

    try:
        # Check if it's an audio or video file
        ext = os.path.splitext(file_path)[1].lower()

        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            # It's an audio file
            audio_paths = [audio_processor.check_audio_file(file_path)]
        else:
            # It's a video file - extract and potentially split
            logger.info("Extracting audio from video...")
            audio_paths = audio_processor.extract_and_split_audio(
                file_path, segment_duration=DEFAULT_SEGMENT_DURATION,  # 10 minute segments
                audio_dir=AUDIO_DIR,
                max_duration=max_duration
            )

        # Process each audio segment
        output_paths = []
        for i, audio_path in enumerate(audio_paths):
            if len(audio_paths) > 1:
                logger.info(
                    "Processing segment %d/%d...",
                    i + 1,
                    len(audio_paths))

            output_path = process_audio_file(
                audio_path, f"{output_name}_seg{i}" if output_name and len(
                    audio_paths) > 1 else output_name,
                whisper_model, whisper_language)

            output_paths.append(output_path)

            # Clean up temporary audio file if it was extracted from video
            if os.path.exists(audio_path) and (
                    ext not in SUPPORTED_AUDIO_EXTENSIONS or 'segment' in audio_path):
                logger.info("Deleting temporary audio file: %s", audio_path)
                try:
                    os.remove(audio_path)
                    logger.debug("Audio file deleted successfully")
                except Exception:  # pylint: disable=broad-except
                    pass

        # If multiple segments, return the first output path
        # TODO: Could combine segments into one file
        return output_paths[0]

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


def process_batch_urls(urls: List[str], output_dir: str = OUTPUT_DIR,  # pylint: disable=unused-argument
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> List[str]:
    """
    Process multiple Bilibili URLs in batch.
    """
    if not urls:
        raise ValueError("URLs list cannot be empty")

    logger.info("Starting batch processing of %d URLs", len(urls))

    output_paths = []
    successful_count = 0
    failed_count = 0

    for i, url in enumerate(urls, 1):
        try:
            logger.info("Processing URL %d/%d: %s", i, len(urls), url)

            # Process single URL
            output_path = process_bilibili_url(
                url, whisper_model=whisper_model,
                whisper_language=whisper_language
            )

            output_paths.append(output_path)
            successful_count += 1

            logger.info("✓ Successfully processed URL %d/%d", i, len(urls))

        except Exception as e:  # pylint: disable=broad-except
            failed_count += 1
            logger.error(
                "✗ Failed to process URL %d/%d: %s",
                i,
                len(urls),
                str(e))

            # Continue processing other URLs even if one fails
            continue

    logger.info("Batch processing complete: %d successful, %d failed",
                successful_count, failed_count)

    return output_paths


def process_batch_files(file_paths: List[str], output_dir: str = OUTPUT_DIR,  # pylint: disable=unused-argument
                        whisper_model: str = WHISPER_MODEL,
                        whisper_language: str = WHISPER_LANGUAGE) -> List[str]:
    """
    Process multiple local files in batch.
    """
    if not file_paths:
        raise ValueError("File paths list cannot be empty")

    logger.info("Starting batch processing of %d files", len(file_paths))

    output_paths = []
    successful_count = 0
    failed_count = 0

    for i, file_path in enumerate(file_paths, 1):
        try:
            logger.info(
                "Processing file %d/%d: %s",
                i,
                len(file_paths),
                file_path)

            # Process single file
            output_path = process_local_file(
                file_path, whisper_model=whisper_model,
                whisper_language=whisper_language,
                split_duration=None  # Use default (7200s)
            )

            output_paths.append(output_path)
            successful_count += 1

            logger.info(
                "✓ Successfully processed file %d/%d",
                i,
                len(file_paths))

        except Exception as e:  # pylint: disable=broad-except
            failed_count += 1
            logger.error(
                "✗ Failed to process file %d/%d: %s",
                i,
                len(file_paths),
                str(e))

            # Continue processing other files even if one fails
            continue

    logger.info("Batch processing complete: %d successful, %d failed",
                successful_count, failed_count)

    return output_paths


def process_audio_file(audio_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> str:
    """
    Process audio file with diarization and transcription.
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    # Step 3: Speaker diarization
    logger.info(
        "Step %d/5: Performing speaker diarization...",
        PIPELINE_STEPS["DIARIZE"])
    diarization_segments = diarizer.diarize_audio(audio_path)

    # Step 4: Transcription
    logger.info(
        "Step %d/5: Transcribing audio...",
        PIPELINE_STEPS["TRANSCRIBE"])
    transcription_segments = transcriber.get_text_with_timestamps(
        audio_path, whisper_model, whisper_language
    )

    # Step 5: Merge and save
    logger.info(
        "Step %d/5: Merging results and formatting output...",
        PIPELINE_STEPS["MERGE"])
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
@click.argument('input_source', required=False)
@click.option('--file', 'is_file', is_flag=True,
              help='Treat input as a file path instead of URL')
@click.option('--batch', 'batch_file',
              help='Process multiple URLs/files from a text file (one per line)')
@click.option('--output-dir', default=OUTPUT_DIR,
              help='Output directory for transcriptions')
@click.option('--model', 'whisper_model', default=WHISPER_MODEL,
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--whisper-language', default=WHISPER_LANGUAGE,
              help='Language code for Whisper (e.g., zh, en, ja)')
@click.option('--split-duration', type=int,
              help='Split audio into segments (seconds). Default: %d (1 hour) for videos longer than 2 hours') % DEFAULT_SPLIT_THRESHOLD_CLI
def main(input_source: Optional[str], is_file: bool, batch_file: Optional[str],
         output_dir: str, whisper_model: str, whisper_language: str,
         split_duration: Optional[int]) -> None:
    """
    Transcribe Bilibili live recordings with speaker diarization.

    \b
    Examples:
        # Single URL
        python main.py https://www.bilibili.com/video/BVxxxxx

        # Single file
        python main.py --file video.mp4

        # File with automatic splitting for videos > 2 hours
        python main.py --file long_video.mp4

        # Manually set split threshold (split if > 1 hour)
        python main.py --file video.mp4 --split-duration 3600

        # Batch processing from file
        python main.py --batch urls.txt

        # Batch processing with model selection
        python main.py --batch urls.txt --model large
    """
    # Update output directory if specified
    if output_dir != OUTPUT_DIR:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Update the config OUTPUT_DIR for this session
        globals()['OUTPUT_DIR'] = output_dir

    try:
        if batch_file:
            # Batch processing mode
            if not os.path.exists(batch_file):
                click.echo(
                    f"Error: Batch file not found: {batch_file}",
                    err=True)
                sys.exit(1)

            # Read URLs/files from batch file
            with open(batch_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                click.echo("Error: Batch file is empty", err=True)
                sys.exit(1)

            click.echo(f"Processing {len(lines)} items from batch file...")

            if is_file:
                # Batch process local files
                output_paths = process_batch_files(
                    lines, output_dir, whisper_model, whisper_language)
            else:
                # Batch process URLs
                output_paths = process_batch_urls(
                    lines, output_dir, whisper_model, whisper_language)

            click.echo("\n✓ Batch processing complete!")
            click.echo(
                f"✓ {
                    len(output_paths)} transcriptions saved to: {output_dir}")

        elif input_source:
            # Single item processing
            if is_file:
                output_path = process_local_file(input_source, whisper_model=whisper_model,
                                                 whisper_language=whisper_language,
                                                 split_duration=split_duration)
            else:
                output_path = process_bilibili_url(input_source, whisper_model=whisper_model,
                                                   whisper_language=whisper_language)

            click.echo(f"\n✓ Transcription saved to: {output_path}")
        else:
            click.echo(
                "Error: No input source provided. Use --help for usage information.",
                err=True)
            sys.exit(1)

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
