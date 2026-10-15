from typing import Any, Tuple

import numpy as np

from nyha_temporal_identification.constants import DEFAULT_NO_TEXT, REMOVED_SENTENCES

ALLOWED_CHARS = ".-/+ "

# NUL byte used as a line-break sentinel.  It never appears in clinical text,
# is not matched by \s in Python's re engine, and is not in [a-z0-9], so it
# acts as a hard word-boundary without being mistaken for a regular space.
LINE_BREAK_SENTINEL = "\x00"


def clean_text(
    original_text: str | list[str] | Any,
    normalize_text: bool = True,
) -> Tuple[str, dict[int, int]]:
    """
    Cleans text by:
    - Lowercasing
    - Removing forbidden sentences
    - Removing non-alphanumeric chars (preserving '.', '-', '/', ' ')
    - Converting newlines / carriage-returns to LINE_BREAK_SENTINEL (\x00)
      so they act as hard word boundaries without being confused with spaces
    - Normalizing runs of whitespace / sentinels to a single character

    Returns:
        Tuple of (cleaned_string, index_mapping)
    """
    # Normalize input type
    if isinstance(original_text, list):
        original_text = " ".join(original_text)
    # Check for numpy array
    elif isinstance(original_text, np.ndarray):
        original_text = " ".join(map(str, np.asarray(original_text).flatten()))

    # Ensure it's a string
    original_text = str(original_text)

    # Early exit on forbidden phrases
    if any(phrase in original_text for phrase in REMOVED_SENTENCES):
        return DEFAULT_NO_TEXT, {}

    # Lowercase
    text = original_text.lower()

    cleaned_chars: list[str] = []
    index_map: dict[int, int] = {}
    prev_was_space = False

    for orig_idx, char in enumerate(text):
        # Skip non-ASCII characters if requested
        if normalize_text and ord(char) > 127:
            prev_was_space = False
            continue

        # Newlines / carriage-returns → sentinel (hard boundary, not a space).
        # Tabs → regular space.
        if char in "\n\r":
            char = LINE_BREAK_SENTINEL
        elif char == "\t":
            char = " "

        # Keep allowed characters and the sentinel only
        if char.isalnum() or char in ALLOWED_CHARS or char == LINE_BREAK_SENTINEL:
            # Collapse consecutive whitespace (spaces or sentinels) to one char
            if char in (" ", LINE_BREAK_SENTINEL):
                if prev_was_space:
                    continue
                prev_was_space = True
            else:
                prev_was_space = False

            new_idx = len(cleaned_chars)
            cleaned_chars.append(char)
            index_map[new_idx] = orig_idx

    # Build cleaned string (no stripping yet)
    cleaned_string = "".join(cleaned_chars)

    # Trim leading/trailing whitespace (spaces and sentinels) and adjust index map
    stripped = cleaned_string.lstrip(" " + LINE_BREAK_SENTINEL)
    left_trim = len(cleaned_string) - len(stripped)
    stripped = cleaned_string.rstrip(" " + LINE_BREAK_SENTINEL)
    right_trim = len(cleaned_string) - len(stripped)

    if left_trim or right_trim:
        # prevent slicing error if string is empty
        end_idx = (
            len(cleaned_string) - right_trim if right_trim > 0 else len(cleaned_string)
        )
        cleaned_string = cleaned_string[left_trim:end_idx]

        index_map = {
            new_idx - left_trim: old_idx
            for new_idx, old_idx in index_map.items()
            if left_trim <= new_idx < len(cleaned_chars) - right_trim
        }

    return cleaned_string, index_map
