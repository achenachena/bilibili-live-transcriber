"""
Video downloader module for fetching videos from supported platforms.
Uses functional programming approach.
"""
import logging
import os
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional

import yt_dlp

from .config import (
    DOWNLOAD_FORMAT_AUDIO_OPTIMIZED,
    DOWNLOAD_FORMAT_FALLBACK,
    DOWNLOAD_FORMAT_FULL_VIDEO,
    DOWNLOAD_MERGE_FORMAT
)

logger = logging.getLogger(__name__)

# Bilibili: experimental CDN hint when not using BBDown (may reduce watermark)
_BILIBILI_UPOS_EXTRACTOR_ARGS: Dict[str, Dict[str, str]] = {
    'bilibili': {
        'upos_host': 'upos-sz-mirrorcoso1.bilivideo.com',
    }
}


@lru_cache(maxsize=1)
def _resolve_bbdown_command() -> Optional[str]:
    """Return BBDown invoker: 'BBDown', 'bbdown', or 'DOTNET_RUN:<csproj path>'."""
    for cmd in ('BBDown', 'bbdown'):
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    bbdown_path = os.path.expanduser('~/repos/BBDown')
    project_path = os.path.join(bbdown_path, 'BBDown/BBDown.csproj')
    if os.path.exists(project_path):
        try:
            result = subprocess.run(
                ['dotnet', 'run', '--project', project_path,
                 '--', '--version'],
                capture_output=True,
                timeout=10,
                check=False
            )
            if result.returncode == 0:
                return f"DOTNET_RUN:{project_path}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None


def _bbdown_subprocess_argv(bbdown_cmd: str, url: str,
                            output_dir: str) -> list[str]:
    """Build argv for BBDown or dotnet run wrapper."""
    if bbdown_cmd.startswith('DOTNET_RUN:'):
        project_path = bbdown_cmd.split(':', 1)[1]
        return [
            'dotnet', 'run', '--project', project_path, '--',
            url, '-tv', '-o', output_dir,
        ]
    return [bbdown_cmd, url, '-tv', '-o', output_dir]


def _is_bbdown_available() -> bool:
    """Check if BBDown is available in the system."""
    return _resolve_bbdown_command() is not None


def _base_ydl_opts(output_dir: str) -> Dict[str, Any]:
    """Shared yt-dlp options except format and extractor_args."""
    return {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'merge_output_format': DOWNLOAD_MERGE_FORMAT,
    }


def _finalize_ytdlp_download(
        info: Any,
        output_dir: str,
        ydl: yt_dlp.YoutubeDL,
        *,
        use_merged_file_heuristic: bool,
        log_downloaded_label: bool = True) -> str:
    """Resolve local media path after yt-dlp extract_info(..., download=True)."""
    if not isinstance(info, dict):
        raise FileNotFoundError("Invalid download info from yt-dlp")

    if info.get('_type') == 'playlist':
        logger.info(
            "Downloaded playlist with %d entries",
            len(info.get('entries', [])))
        downloaded_files = _get_recent_files(output_dir)
        if downloaded_files:
            filename = downloaded_files[0]
            logger.info("Using first playlist entry: %s", filename)
            return filename
        raise FileNotFoundError("No files downloaded from playlist")

    if use_merged_file_heuristic:
        filename = _find_downloaded_file_by_info(info, output_dir)
        if not filename:
            filename = ydl.prepare_filename(info)
            if not os.path.exists(filename):
                filename = _find_downloaded_file(filename)
                if not filename:
                    downloaded_files = _get_recent_files(output_dir)
                    if downloaded_files:
                        filename = downloaded_files[0]
                        logger.info(
                            "Using most recently downloaded file: %s",
                            filename)
                    else:
                        raise FileNotFoundError(
                            f"Downloaded file not found: {filename}")
    else:
        filename = ydl.prepare_filename(info)
        if log_downloaded_label:
            logger.info("Downloaded: %s", filename)
        if not os.path.exists(filename):
            filename = _find_downloaded_file(filename)
            if not filename:
                raise FileNotFoundError(
                    f"Downloaded file not found: {filename}")

    if use_merged_file_heuristic and log_downloaded_label:
        logger.info("Downloaded: %s", filename)
    return filename


def detect_video_platform(url: str) -> str:
    """Detect video platform from URL."""
    if not url:
        return "unknown"

    url_lower = url.lower()
    if 'bilibili.com' in url_lower or 'b23.tv' in url_lower:
        return "Bilibili"
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return "YouTube"
    return "video platform"


def download_video(url: str, output_dir: str = "videos") -> str:
    """Download a video from a supported platform URL and return the file path."""
    if not url:
        raise ValueError("URL cannot be empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    platform = detect_video_platform(url)
    logger.info("Starting download from %s: %s", platform, url)
    logger.info(
        "Using audio-optimized format: %s",
        DOWNLOAD_FORMAT_AUDIO_OPTIMIZED)

    ydl_opts: Dict[str, Any] = {
        **_base_ydl_opts(output_dir),
        'format': DOWNLOAD_FORMAT_AUDIO_OPTIMIZED,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return _finalize_ytdlp_download(
                info, output_dir, ydl, use_merged_file_heuristic=False)

    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Audio-optimized download failed: %s", str(e))
        logger.info("Trying fallback format: %s", DOWNLOAD_FORMAT_FALLBACK)

        ydl_opts_fallback = {
            **_base_ydl_opts(output_dir),
            'format': DOWNLOAD_FORMAT_FALLBACK,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                info = ydl.extract_info(url, download=True)
                try:
                    path = _finalize_ytdlp_download(
                        info, output_dir, ydl,
                        use_merged_file_heuristic=False,
                        log_downloaded_label=False)
                except FileNotFoundError as fnf:
                    raise fnf from e
                logger.info("Fallback download successful: %s", path)
                return path

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
    # Remove format codes like .f30066 that yt-dlp adds
    base_name = base_name.split('.f')[0]
    for ext in ['.mp4', '.mkv', '.webm', '.flv']:
        candidate = base_name + ext
        if os.path.exists(candidate):
            return candidate
    return None


def _download_with_bbdown(url: str, output_dir: str, bbdown_cmd: str) -> str:
    """Download video using BBDown (better watermark removal for Bilibili).

    Args:
        url: Video URL to download
        output_dir: Directory to save the video
        bbdown_cmd: BBDown command name ('BBDown', 'bbdown', or 'DOTNET_RUN:path')

    Returns:
        Path to downloaded video file

    Raises:
        RuntimeError: If BBDown download fails
    """
    logger.info("Using BBDown to download: %s", url)

    # BBDown command: BBDown <url> -tv -o <output_dir>
    # -tv flag downloads TV version (no watermark)
    # -o specifies output directory
    # Note: URL must be the first argument after the command
    try:
        cmd = _bbdown_subprocess_argv(bbdown_cmd, url, output_dir)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout
        )

        logger.info("BBDown download completed successfully")
        logger.debug("BBDown output: %s", result.stdout)

        # BBDown saves files with specific naming, find the downloaded file
        # It typically saves as: <title>.mp4
        downloaded_files = _get_recent_files(output_dir)
        if downloaded_files:
            filename = downloaded_files[0]
            logger.info("BBDown downloaded: %s", filename)
            return filename
        raise RuntimeError(
            "BBDown completed but no video file found in output directory")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"BBDown download failed: {error_msg}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("BBDown download timed out") from None


def _find_downloaded_file_by_info(
        info: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Find downloaded file based on video info, handling merged files."""
    if not info:
        return None

    expected_filename: Optional[str] = None
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            expected_filename = ydl.prepare_filename(info)
    except Exception:  # pylint: disable=broad-except
        expected_filename = None

    # First try the expected filename
    if expected_filename and os.path.exists(expected_filename):
        return expected_filename

    # Try without format codes
    if expected_filename:
        found = _find_downloaded_file(expected_filename)
        if found:
            return found

    # Get title and search for files with similar name
    title = info.get('title', '')
    if title:
        # Find files that match the title
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath) and filename.endswith(
                        ('.mp4', '.mkv', '.webm', '.flv')):
                    # Check if filename contains title (allowing for some
                    # variation)
                    if title in filename or filename.startswith(title[:20]):
                        return filepath

    # Last resort: return most recent file
    downloaded_files = _get_recent_files(output_dir)
    if downloaded_files:
        return downloaded_files[0]

    return None


def download_full_video(url: str, output_dir: str = "videos",
                        no_watermark: bool = True) -> str:
    """Download a full video (best quality) from a supported platform URL and return the file path.

    Args:
        url: Video URL to download
        output_dir: Directory to save the video
        no_watermark: For Bilibili, download without UP主 watermark (default: True)
                     If True and BBDown is available, will use BBDown for better watermark removal
    """
    if not url:
        raise ValueError("URL cannot be empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    platform = detect_video_platform(url)

    # For Bilibili, try BBDown first if available and no_watermark is True
    if no_watermark and platform == "Bilibili":
        bbdown_cmd = _resolve_bbdown_command()
        if bbdown_cmd:
            logger.info(
                "BBDown detected. Using BBDown for better watermark removal.")
            try:
                return _download_with_bbdown(url, output_dir, bbdown_cmd)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "BBDown download failed: %s. Falling back to yt-dlp.", str(e))
                # Fall through to yt-dlp

    logger.info("Starting full video download from %s: %s", platform, url)
    logger.info("Using full video format: %s", DOWNLOAD_FORMAT_FULL_VIDEO)
    if no_watermark and platform == "Bilibili":
        logger.info("Downloading without UP主 watermark")
        if not _is_bbdown_available():
            logger.warning(
                "Note: yt-dlp may not fully remove Bilibili watermarks. "
                "For better watermark removal, install BBDown:")
            logger.warning(
                "  1. Download from: https://github.com/nilaoda/BBDown/releases")
            logger.warning(
                "  2. Extract and add to PATH, or use full path to BBDown")
            logger.warning(
                "  3. BBDown supports -tv flag for TV version (no watermark)")

    ydl_opts: Dict[str, Any] = {
        **_base_ydl_opts(output_dir),
        'format': DOWNLOAD_FORMAT_FULL_VIDEO,
    }

    if no_watermark and platform == "Bilibili":
        ydl_opts['extractor_args'] = dict(_BILIBILI_UPOS_EXTRACTOR_ARGS)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return _finalize_ytdlp_download(
                info, output_dir, ydl, use_merged_file_heuristic=True)

    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Full video download failed: %s", str(e))
        logger.info("Trying fallback format: %s", DOWNLOAD_FORMAT_FALLBACK)

        ydl_opts_fallback: Dict[str, Any] = {
            **_base_ydl_opts(output_dir),
            'format': DOWNLOAD_FORMAT_FALLBACK,
        }

        if no_watermark and platform == "Bilibili":
            ydl_opts_fallback['extractor_args'] = dict(
                _BILIBILI_UPOS_EXTRACTOR_ARGS)

        try:
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                info = ydl.extract_info(url, download=True)
                try:
                    path = _finalize_ytdlp_download(
                        info, output_dir, ydl,
                        use_merged_file_heuristic=True,
                        log_downloaded_label=False)
                except FileNotFoundError as fnf:
                    raise fnf from e
                logger.info("Fallback download successful: %s", path)
                return path

        except Exception as fallback_error:
            logger.error("Both full video and fallback downloads failed")
            logger.error("Full video error: %s", str(e))
            logger.error("Fallback error: %s", str(fallback_error))
            raise


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
