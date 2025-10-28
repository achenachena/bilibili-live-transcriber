"""
Module to merge diarization and transcription results.
Uses functional programming approach.
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def merge_results(diarization_segments: List[Tuple[float, float, str]],
                  transcription_segments: List[Tuple[float, float, str]],
                  min_speaker_duration: float = 0.5,
                  min_silence_duration: float = 0.5) -> List[Tuple[float, float, str, str]]:
    """
    Merge diarization and transcription segments.

    Uses RORO pattern: Receive an Object, Return an Object

    Args:
        diarization_segments: List of (start, end, speaker) tuples
        transcription_segments: List of (start, end, text) tuples
        min_speaker_duration: Minimum duration for a speaker segment (seconds)
        min_silence_duration: Minimum silence duration to split segments (seconds)

    Returns:
        List of (start, end, speaker, text) tuples
    """
    if not diarization_segments:
        logger.warning("No diarization segments provided")
        return []

    if not transcription_segments:
        logger.warning("No transcription segments provided")
        return []

    logger.info("Merging diarization and transcription results...")

    merged = []

    for trans_start, trans_end, text in transcription_segments:
        speaker = _find_speaker_for_time(
            diarization_segments, trans_start, trans_end)

        if speaker:
            merged.append((trans_start, trans_end, speaker, text))

    # Group consecutive segments from the same speaker
    merged = _group_consecutive_segments(
        merged, min_speaker_duration, min_silence_duration)

    logger.info("Merged %d segments", len(merged))
    return merged


def _find_speaker_for_time(diarization_segments: List[Tuple[float, float, str]],
                           start_time: float, end_time: float) -> Optional[str]:
    """
    Find which speaker is active during a given time range.

    Internal helper function.

    Args:
        diarization_segments: List of (start, end, speaker) tuples
        start_time: Start time of the segment
        end_time: End time of the segment

    Returns:
        Speaker label or None
    """
    overlaps = []

    for diar_start, diar_end, speaker in diarization_segments:
        overlap_start = max(start_time, diar_start)
        overlap_end = min(end_time, diar_end)

        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            segment_duration = end_time - start_time
            overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0

            overlaps.append((overlap_ratio, speaker))

    if overlaps:
        overlaps.sort(reverse=True)
        return overlaps[0][1]

    return None


def _group_consecutive_segments(segments: List[Tuple[float, float, str, str]],
                                min_speaker_duration: float,
                                min_silence_duration: float) -> List[Tuple[float, float, str, str]]:
    """
    Group consecutive segments from the same speaker.

    Internal helper function.

    Args:
        segments: List of (start, end, speaker, text) tuples
        min_speaker_duration: Minimum duration for a speaker segment
        min_silence_duration: Minimum silence duration to split segments

    Returns:
        Grouped segments
    """
    if not segments:
        return segments

    grouped = []
    current_start, current_end, current_speaker, current_text = segments[0]

    for start, end, speaker, text in segments[1:]:
        # Check if same speaker and close enough
        if speaker == current_speaker and start - current_end <= min_silence_duration:
            # Merge with current segment
            current_end = end
            current_text += " " + text
        else:
            # Save current segment and start new one
            if current_end - current_start >= min_speaker_duration:
                grouped.append((current_start, current_end,
                               current_speaker, current_text))
            current_start, current_end, current_speaker, current_text = start, end, speaker, text

    # Add the last segment
    if current_end - current_start >= min_speaker_duration:
        grouped.append((current_start, current_end,
                       current_speaker, current_text))

    return grouped
