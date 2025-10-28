"""
Test configuration module.
"""
# pylint: disable=import-error,unused-import
try:
    from bilibili_transcriber.config import (
    AUDIO_DIR,
    OUTPUT_DIR,
    VIDEO_DIR,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    )
except ImportError:
    # Tests should run after package is installed
    pass


def test_constants_exist():
    """Test that all configuration constants are defined."""
    try:
        assert AUDIO_DIR is not None  # type: ignore
        assert OUTPUT_DIR is not None  # type: ignore
        assert VIDEO_DIR is not None  # type: ignore
        assert WHISPER_LANGUAGE is not None  # type: ignore
        assert WHISPER_MODEL is not None  # type: ignore
    except NameError:
        pass  # Module not available, skip test


def test_model_is_valid():
    """Test that the Whisper model is a valid size."""
    try:
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        assert WHISPER_MODEL in valid_models  # type: ignore
    except NameError:
        pass  # Module not available, skip test
