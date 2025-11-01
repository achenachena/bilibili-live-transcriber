"""
Video downloader module for fetching Bilibili recordings.
Uses functional programming approach.
"""
import logging
import os
from typing import Any, Dict, Optional

import yt_dlp

from .config import (
    DOWNLOAD_FORMAT_AUDIO_OPTIMIZED,
    DOWNLOAD_FORMAT_FALLBACK,
    DOWNLOAD_MERGE_FORMAT
)

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str = "videos") -> str:
    """Download a video from a Bilibili URL and return the file path."""
    if not url:
        raise ValueError("URL cannot be empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting download from: %s", url)
    logger.info(
        "Using audio-optimized format: %s",
        DOWNLOAD_FORMAT_AUDIO_OPTIMIZED)

    # Try audio-optimized format first
    ydl_opts = {
        'format': DOWNLOAD_FORMAT_AUDIO_OPTIMIZED,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'merge_output_format': DOWNLOAD_MERGE_FORMAT,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Handle playlist downloads
            if isinstance(info, dict) and info.get('_type') == 'playlist':
                logger.info(
                    "Downloaded playlist with %d entries", len(
                        info.get(
                            'entries', [])))

                # For playlists, find the most recently downloaded file
                # by checking the directory for new files
                downloaded_files = _get_recent_files(output_dir)

                if downloaded_files:
                    # Return the first video file
                    filename = downloaded_files[0]
                    logger.info("Using first playlist entry: %s", filename)
                    return filename
                else:
                    raise FileNotFoundError(
                        "No files downloaded from playlist")
            else:
                # Single video download
                filename = ydl.prepare_filename(info)
                logger.info("Downloaded: %s", filename)

                # Ensure file exists
                if not os.path.exists(filename):
                    filename = _find_downloaded_file(filename)
                    if not filename:
                        raise FileNotFoundError(
                            f"Downloaded file not found: {filename}")

                return filename

    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Audio-optimized download failed: %s", str(e))
        logger.info("Trying fallback format: %s", DOWNLOAD_FORMAT_FALLBACK)

        # Fallback to standard format
        ydl_opts_fallback = {
            'format': DOWNLOAD_FORMAT_FALLBACK,
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'merge_output_format': DOWNLOAD_MERGE_FORMAT,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                info = ydl.extract_info(url, download=True)

                if isinstance(info, dict) and info.get('_type') == 'playlist':
                    downloaded_files = _get_recent_files(output_dir)
                    if downloaded_files:
                        filename = downloaded_files[0]
                        logger.info(
                            "Fallback download successful: %s", filename)
                        return filename
                    else:
                        raise FileNotFoundError(
                            "No files downloaded from playlist") from e
                else:
                    filename = ydl.prepare_filename(info)
                    logger.info("Fallback download successful: %s", filename)

                    if not os.path.exists(filename):
                        filename = _find_downloaded_file(filename)
                        if not filename:
                            raise FileNotFoundError(
                                f"Downloaded file not found: {filename}") from e

                    return filename

        except Exception as fallback_error:
            logger.error("Both audio-optimized and fallback downloads failed")
            logger.error("Audio-optimized error: %s", str(e))
            logger.error("Fallback error: %s", str(fallback_error))
            raise


def _get_recent_files(directory: str) -> list[str]:
    """Return most recent downloaded video files in a directory."""
    if not os.path.exists(directory):
        return []

    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(
                ('.mp4', '.mkv', '.webm', '.flv')):
            files.append((os.path.getmtime(filepath), filepath))

    # Sort by modification time, newest first
    files.sort(reverse=True)
    return [filepath for _mtime, filepath in files]


def _find_downloaded_file(filename: str) -> Optional[str]:
    """Return existing file by trying common video extensions for a base name."""
    base_name = os.path.splitext(filename)[0]
    for ext in ['.mp4', '.mkv', '.webm', '.flv']:
        candidate = base_name + ext
        if os.path.exists(candidate):
            return candidate
    return None


def get_video_info(url: str) -> Dict[str, Any]:
    """Return video information without downloading."""
    if not url:
        raise ValueError("URL cannot be empty")

    ydl_opts = {'quiet': True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        logger.error("Failed to get video info: %s", str(e), exc_info=True)
        raise
