"""
Core ML model conversion for Apple Silicon Neural Engine acceleration.
Converts PyTorch models to Core ML format for optimized inference on Apple Silicon.
"""
import logging
import os
import importlib
from typing import Optional

logger = logging.getLogger(__name__)

# Core ML model cache
_CORE_ML_MODEL_CACHE = {}


def convert_whisper_to_coreml(
    _whisper_model,
    model_size: str,
    cache_dir: str = ".coreml_cache"
) -> Optional[object]:
    """Convert Whisper to Core ML and return the model or None."""
    global _CORE_ML_MODEL_CACHE  # pylint: disable=global-statement,global-variable-not-assigned

    cache_key = f"whisper_{model_size}_coreml"
    if cache_key in _CORE_ML_MODEL_CACHE:
        logger.info("Using cached Core ML Whisper model")
        return _CORE_ML_MODEL_CACHE[cache_key]

    try:

        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, f"whisper_{model_size}.mlmodel")

        # Check if cached model exists
        if os.path.exists(model_path):
            logger.info("Loading cached Core ML model from %s", model_path)
            try:
                coremltools_mod = importlib.import_module(
                    "coremltools")  # type: ignore[import-not-found]
                coreml_model = coremltools_mod.models.MLModel(model_path)
                _CORE_ML_MODEL_CACHE[cache_key] = coreml_model
                logger.info("Core ML model loaded successfully")
                return coreml_model
            except (OSError, RuntimeError, ValueError) as load_error:
                logger.warning(
                    "Failed to load cached Core ML model: %s. Re-converting...",
                    str(load_error)
                )

        logger.info("Converting Whisper model to Core ML format...")
        logger.info("This may take several minutes...")

        # Note: Full Whisper model conversion is complex
        # This is a placeholder for future implementation
        # Core ML conversion requires tracing and quantization
        logger.warning(
            "Core ML conversion for Whisper is experimental and not fully implemented."
        )
        logger.warning("Falling back to PyTorch with MPS acceleration.")
        return None

    except ImportError:
        logger.info("Core ML tools not available")
        return None
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to convert Whisper to Core ML: %s", str(e))
        return None


def convert_pyannote_to_coreml(
    _pyannote_pipeline,
    cache_dir: str = ".coreml_cache"
) -> Optional[object]:
    """Convert pyannote pipeline to Core ML and return it or None."""
    global _CORE_ML_MODEL_CACHE  # pylint: disable=global-statement,global-variable-not-assigned

    cache_key = "pyannote_coreml"
    if cache_key in _CORE_ML_MODEL_CACHE:
        logger.info("Using cached Core ML pyannote pipeline")
        return _CORE_ML_MODEL_CACHE[cache_key]

    try:

        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "pyannote.mlmodel")

        # Check if cached model exists
        if os.path.exists(model_path):
            logger.info("Loading cached Core ML model from %s", model_path)
            try:
                coremltools_mod = importlib.import_module(
                    "coremltools")  # type: ignore[import-not-found]
                coreml_model = coremltools_mod.models.MLModel(model_path)
                _CORE_ML_MODEL_CACHE[cache_key] = coreml_model
                logger.info("Core ML pipeline loaded successfully")
                return coreml_model
            except (OSError, RuntimeError, ValueError) as load_error:
                logger.warning(
                    "Failed to load cached Core ML pipeline: %s. Re-converting...",
                    str(load_error)
                )

        logger.info("Converting pyannote pipeline to Core ML format...")

        # Note: Full pyannote conversion is extremely complex
        # This is a placeholder for future implementation
        logger.warning(
            "Core ML conversion for pyannote is experimental and not fully implemented."
        )
        logger.warning("Falling back to PyTorch with MPS acceleration.")
        return None

    except ImportError:
        logger.info("Core ML tools not available")
        return None
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to convert pyannote to Core ML: %s", str(e))
        return None


def is_coreml_conversion_ready() -> bool:
    """Return True if Core ML tools are available."""
    try:
        return importlib.util.find_spec("coremltools") is not None
    except Exception:  # pylint: disable=broad-except
        return False
