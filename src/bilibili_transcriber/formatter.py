"""
Output formatting module for generating text files.
Uses functional programming approach.
"""
import logging
import os
from datetime import timedelta
from typing import List, Tuple

from .config import SECONDS_PER_HOUR, SECONDS_PER_MINUTE, PUNCTUATION_RULES

logger = logging.getLogger(__name__)


def _add_commas(text: str, language: str = 'zh') -> str:
    """Insert commas to improve readability based on language."""
    if language == 'zh':
        # First, replace spaces with commas
        result = text.replace(' ', '，')

        # Then clean up conflicts: remove commas that are followed by other
        # punctuation
        import re
        # Remove comma followed by period, exclamation, or question mark
        result = re.sub(r'，([。！？])', r'\1', result)

        return result
    else:
        # For English, keep spaces but add commas before conjunctions
        replacements = [
            (' and', ', and'),
            (' but', ', but'),
            (' so', ', so'),
            (' then', ', then'),
        ]

        result = text
        for old, new in replacements:
            result = result.replace(old, new)

        return result


def restore_punctuation(text: str, language: str = 'zh') -> str:
    """Restore punctuation in transcribed text using simple rules."""
    if not text or not text.strip():
        return text

    # Get language-specific rules
    rules = PUNCTUATION_RULES.get(language, PUNCTUATION_RULES['zh'])

    # Clean the text first
    text = text.strip()

    # Add commas for natural pauses and breaks
    text = _add_commas(text, language)

    # Split into sentences based on natural breaks
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char

        # Check for natural sentence breaks
        if char in rules['sentence_endings']:
            sentences.append(current_sentence.strip())
            current_sentence = ""
        elif len(current_sentence) > 20 and char in ['，', ',', '、']:
            # Add comma for natural pauses
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # Add remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        if not sentence:
            continue

        # Add appropriate ending punctuation
        sentence = sentence.strip()
        if not any(sentence.endswith(p) for p in rules['sentence_endings']):
            # Check if it's a question
            if any(word in sentence for word in rules['question_words']):
                sentence += '？' if language == 'zh' else '?'
            # Check if it's an exclamation
            elif any(word in sentence for word in rules['exclamation_words']):
                sentence += '！' if language == 'zh' else '!'
            else:
                sentence += '。' if language == 'zh' else '.'

        processed_sentences.append(sentence)

    result = ' '.join(processed_sentences) if language == 'en' else ''.join(
        processed_sentences)

    # Final cleanup: remove commas that are followed by other punctuation
    if language == 'zh':
        import re
        result = re.sub(r'，([。！？])', r'\1', result)

    return result


def convert_to_simplified(text: str) -> str:
    """
    Convert traditional Chinese to simplified Chinese.
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
    """Return time string [HH:MM:SS] for given seconds."""
    td = timedelta(seconds=float(seconds))
    hours, remainder = divmod(td.seconds, SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, SECONDS_PER_MINUTE)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_segments(segments: List[Tuple[float, float, str, str]]) -> str:
    """Format merged segments as Markdown text."""
    lines = ["# Transcription", ""]

    for start, _end, speaker, text in segments:
        # Convert to simplified Chinese
        simplified_text = convert_to_simplified(text)

        # Restore punctuation
        punctuated_text = restore_punctuation(simplified_text, 'zh')

        time_str = format_time(start)
        lines.append(f"**[{time_str}] {speaker}:** {punctuated_text}")
        lines.append("")  # Add blank line for readability

    return "\n".join(lines)


def save_transcription(segments: List[Tuple[float, float, str, str]],
                       output_filename: str, output_dir: str = "output") -> str:
    """Save formatted segments to a file and return the path."""
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


def generate_filename(source_name: str, extension: str = ".md") -> str:
    """Generate an output filename based on the source name."""
    if not source_name:
        return f"transcript{extension}"

    if os.path.exists(source_name):
        base_name = os.path.splitext(os.path.basename(source_name))[0]
    else:
        base_name = source_name.replace('/', '_').replace(':', '_')

    return f"{base_name}_transcript{extension}"
