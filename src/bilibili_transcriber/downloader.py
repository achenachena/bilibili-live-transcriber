"""
Video downloader module for fetching Bilibili recordings.
Uses functional programming approach.
"""
import logging
import os
from typing import Any, Dict, Optional

import yt_dlp

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str = "videos") -> str:
    """
    Download a video from the given Bilibili URL.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        url: Bilibili video URL
        output_dir: Directory to save the video

    Returns:
        Path to the downloaded video file

    Raises:
        FileNotFoundError: If downloaded file cannot be found
        Exception: If download fails
    """
    if not url:
        raise ValueError("URL cannot be empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting download from: %s", url)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'merge_output_format': 'mp4',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            logger.info("Downloaded: %s", filename)

            # Ensure file exists
            if not os.path.exists(filename):
                filename = _find_downloaded_file(filename)
                if not filename:
                    raise FileNotFoundError(
                        f"Downloaded file not found: {filename}")

            return filename

    except Exception as e:
        logger.error("Download failed: %s", str(e), exc_info=True)
        raise


def _find_downloaded_file(filename: str) -> Optional[str]:
    """
    Find the actual downloaded file by trying different extensions.

    Internal helper function.

    Args:
        filename: Base filename without extension

    Returns:
        Full path to found file, or None
    """
    base_name = os.path.splitext(filename)[0]
    for ext in ['.mp4', '.mkv', '.webm', '.flv']:
        candidate = base_name + ext
        if os.path.exists(candidate):
            return candidate
    return None


def get_video_info(url: str) -> Dict[str, Any]:
    """
    Get video information without downloading.

    Args:
        url: Bilibili video URL

    Returns:
        Video info dictionary

    Raises:
        Exception: If unable to extract video info
    """
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
