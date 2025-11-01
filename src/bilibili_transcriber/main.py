"""
Main CLI interface for Bilibili Live Transcriber.
Uses functional programming approach.
"""
import logging
import os
import sys
from contextlib import contextmanager
from typing import Generator, List, Optional
import concurrent.futures
import time

import click

from . import (audio_processor, diarizer, downloader, formatter, merger,
               transcriber)
from .config import (AUDIO_DIR, OUTPUT_DIR, VIDEO_DIR, WHISPER_LANGUAGE,
                     WHISPER_MODEL, PIPELINE_STEPS, DEFAULT_SPLIT_THRESHOLD,
                     DEFAULT_SPLIT_THRESHOLD_CLI, DEFAULT_SEGMENT_DURATION,
                     SUPPORTED_AUDIO_EXTENSIONS, get_optimal_workers,
                     should_use_parallel_processing, MAX_BATCH_WORKERS)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def managed_file_cleanup(*file_paths: str) -> Generator[List[str], None, None]:
    """Context manager that deletes specified files on exit."""
    try:
        yield list(file_paths)
    finally:
        # Guaranteed cleanup - runs regardless of success/failure
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    logger.info("Deleting temporary file: %s", file_path)
                    os.remove(file_path)
                    logger.debug("File deleted successfully: %s", file_path)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to delete file %s: %s", file_path, e)


def process_bilibili_url(url: str, output_name: Optional[str] = None,
                         whisper_model: str = WHISPER_MODEL,
                         whisper_language: str = WHISPER_LANGUAGE) -> str:
    """Process a Bilibili URL through the full pipeline."""
    if not url:
        raise ValueError("URL cannot be empty")

    logger.info("Processing Bilibili URL: %s", url)

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

    # Use context manager for guaranteed cleanup
    with managed_file_cleanup(video_path, audio_path):
        # Step 3-5: Process audio
        output_path = process_audio_file(
            audio_path, output_name, whisper_model, whisper_language)
        return output_path


def process_local_file(file_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE,
                       split_duration: Optional[int] = None) -> str:
    """Process a local file through the pipeline (splits long videos)."""
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
        temp_audio_files = []  # Track files to clean up

        # Determine if we should use parallel processing
        use_parallel = should_use_parallel_processing(len(audio_paths))

        if use_parallel:
            logger.info(
                "Processing %d segments in parallel...",
                len(audio_paths))
            output_paths = _process_segments_parallel(
                audio_paths, output_name, whisper_model, whisper_language)
        else:
            logger.info(
                "Processing %d segments sequentially...",
                len(audio_paths))
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

        # Track temporary files for cleanup
        for audio_path in audio_paths:
            if ext not in SUPPORTED_AUDIO_EXTENSIONS or 'segment' in audio_path:
                temp_audio_files.append(audio_path)

        # Use context manager for guaranteed cleanup of temporary files
        with managed_file_cleanup(*temp_audio_files):
            # If multiple segments, return the first output path
            return output_paths[0]

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


def process_batch_urls(urls: List[str], output_dir: str = OUTPUT_DIR,  # pylint: disable=unused-argument
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> List[str]:
    """Process multiple Bilibili URLs in batch."""
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


def _process_batch_urls_parallel(
        urls: List[str], whisper_model: str, whisper_language: str) -> List[str]:
    """Process multiple Bilibili URLs in parallel."""
    if not urls:
        return []

    # Use limited workers for batch processing to avoid overwhelming system
    max_workers = min(len(urls), MAX_BATCH_WORKERS)

    logger.info(
        "Processing %d URLs in parallel with %d workers",
        len(urls),
        max_workers)

    def process_single_url(args):
        """Process a single URL."""
        i, url = args
        try:
            logger.info(
                "Starting parallel processing of URL %d/%d: %s",
                i + 1,
                len(urls),
                url)
            start_time = time.time()

            output_path = process_bilibili_url(
                url, whisper_model=whisper_model, whisper_language=whisper_language)

            elapsed = time.time() - start_time
            logger.info(
                "✓ Completed URL %d/%d in %.2f seconds",
                i + 1,
                len(urls),
                elapsed)

            return output_path
        except Exception as e:
            logger.error(
                "✗ Failed to process URL %d/%d: %s",
                i + 1,
                len(urls),
                str(e))
            raise

    # Process URLs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_url, (i, url)): i
            for i, url in enumerate(urls)
        }

        # Collect results
        output_paths = []
        successful_count = 0
        failed_count = 0

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                output_path = future.result()
                output_paths.append(output_path)
                successful_count += 1
            except Exception as e:  # pylint: disable=broad-except
                failed_count += 1
                logger.error("URL %d failed: %s", index + 1, str(e))
                # Continue processing other URLs

    logger.info(
        "Parallel batch processing complete: %d successful, %d failed",
        successful_count,
        failed_count)
    return output_paths


def _process_batch_files_parallel(
        file_paths: List[str], whisper_model: str, whisper_language: str) -> List[str]:
    """Process multiple local files in parallel."""
    if not file_paths:
        return []

    # Use limited workers for batch processing to avoid overwhelming system
    max_workers = min(len(file_paths), MAX_BATCH_WORKERS)

    logger.info(
        "Processing %d files in parallel with %d workers",
        len(file_paths),
        max_workers)

    def process_single_file(args):
        """Process a single file."""
        i, file_path = args
        try:
            logger.info(
                "Starting parallel processing of file %d/%d: %s",
                i + 1,
                len(file_paths),
                file_path)
            start_time = time.time()

            output_path = process_local_file(
                file_path,
                whisper_model=whisper_model,
                whisper_language=whisper_language)

            elapsed = time.time() - start_time
            logger.info(
                "✓ Completed file %d/%d in %.2f seconds",
                i + 1,
                len(file_paths),
                elapsed)

            return output_path
        except Exception as e:
            logger.error(
                "✗ Failed to process file %d/%d: %s",
                i + 1,
                len(file_paths),
                str(e))
            raise

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_file, (i, file_path)): i
            for i, file_path in enumerate(file_paths)
        }

        # Collect results
        output_paths = []
        successful_count = 0
        failed_count = 0

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                output_path = future.result()
                output_paths.append(output_path)
                successful_count += 1
            except Exception as e:  # pylint: disable=broad-except
                failed_count += 1
                logger.error("File %d failed: %s", index + 1, str(e))
                # Continue processing other files

    logger.info(
        "Parallel batch processing complete: %d successful, %d failed",
        successful_count,
        failed_count)
    return output_paths


def process_batch_files(file_paths: List[str], output_dir: str = OUTPUT_DIR,  # pylint: disable=unused-argument
                        whisper_model: str = WHISPER_MODEL,
                        whisper_language: str = WHISPER_LANGUAGE) -> List[str]:
    """Process multiple local files in batch."""
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


def _process_segments_parallel(audio_paths: List[str], output_name: Optional[str],
                               whisper_model: str, whisper_language: str) -> List[str]:
    """Process multiple audio segments in parallel and return output paths."""
    if not audio_paths:
        return []

    # Calculate optimal number of workers
    max_workers = get_optimal_workers(len(audio_paths))

    logger.info(
        "Using %d parallel workers for %d segments",
        max_workers,
        len(audio_paths))

    def process_single_segment(args):
        """Process a single audio segment."""
        i, audio_path = args
        segment_name = f"{output_name}_seg{i}" if output_name and len(
            audio_paths) > 1 else output_name

        try:
            logger.info(
                "Starting parallel processing of segment %d/%d",
                i + 1,
                len(audio_paths))
            start_time = time.time()

            output_path = process_audio_file(
                audio_path, segment_name, whisper_model, whisper_language)

            elapsed = time.time() - start_time
            logger.info(
                "Completed segment %d/%d in %.2f seconds",
                i + 1,
                len(audio_paths),
                elapsed)

            return output_path
        except Exception as e:
            logger.error("Failed to process segment %d: %s", i + 1, str(e))
            raise

    # Process segments in parallel using processes to bypass GIL
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_segment, (i, audio_path)): i
            for i, audio_path in enumerate(audio_paths)
        }

        # Collect results in order
        output_paths = [None] * len(audio_paths)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                output_paths[index] = future.result()
            except Exception as e:
                logger.error("Segment %d failed: %s", index + 1, str(e))
                raise

    return output_paths


def process_audio_file(audio_path: str, output_name: Optional[str] = None,
                       whisper_model: str = WHISPER_MODEL,
                       whisper_language: str = WHISPER_LANGUAGE) -> str:
    """
    Process audio file with diarization and transcription.
    """
    if not audio_path:
        raise ValueError("Audio path cannot be empty")

    # Step 3 & 4: Perform diarization and transcription in parallel
    logger.info(
        "Steps %d-4/5: Performing speaker diarization and transcription in parallel...",
        PIPELINE_STEPS["DIARIZE"])

    diarization_segments = None
    transcription_segments = None

    def run_diarization():
        """Run speaker diarization."""
        nonlocal diarization_segments
        diarization_segments = diarizer.diarize_audio(audio_path)

    def run_transcription():
        """Run transcription."""
        nonlocal transcription_segments
        transcription_segments = transcriber.get_text_with_timestamps(
            audio_path, whisper_model, whisper_language
        )

    # Run both tasks in parallel using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_diarization),
            executor.submit(run_transcription)
        ]
        concurrent.futures.wait(futures)

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
              help=f'Split audio into segments (seconds). '
              f'Default: {DEFAULT_SPLIT_THRESHOLD_CLI} (1 hour) '
              f'for videos longer than 2 hours')
@click.option('--parallel/--no-parallel', default=True,
              help='Enable parallel processing for multiple segments/files')
@click.option('--max-workers', type=int, default=None,
              help='Maximum number of parallel workers (default: auto-detect)')
def main(input_source: Optional[str], is_file: bool, batch_file: Optional[str],
         output_dir: str, whisper_model: str, whisper_language: str,
         split_duration: Optional[int], parallel: bool, max_workers: Optional[int]) -> None:  # pylint: disable=unused-argument
    """CLI entry: transcribe Bilibili recordings with diarization."""
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
