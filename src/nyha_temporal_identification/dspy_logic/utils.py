from typing import Optional

from nyha_temporal_identification.dspy_logic.regex import CLASSIFICATION_CONTEXT_WINDOW

SPAN_MARKER = ">>>"
SPAN_MARKER_END = "<<<"


def truncate_to_token_budget(
    text: str,
    tokenizer,
    max_tokens: int,
    span_open: str = SPAN_MARKER,
    span_close: str = SPAN_MARKER_END,
) -> str:
    """Enforce a hard token ceiling on text using the HF tokenizer.

    Returns text unchanged when tokenizer is None, max_tokens <= 0, or already
    within budget. When truncating: if span markers are present, trims
    symmetrically from both ends so the >>> ... <<< pair is always preserved;
    otherwise truncates from the tail.

    Uses offset_mapping to find exact character boundaries so the returned
    string is always a verbatim slice of the input (no tokenizer-decode
    artefacts that would shift character positions).
    """
    if tokenizer is None or max_tokens <= 0:
        return text

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)

    if n <= max_tokens:
        return text

    open_idx = text.find(span_open)
    close_idx = text.rfind(span_close)

    if open_idx == -1 or close_idx == -1 or open_idx > close_idx:
        # No valid markers: tail truncation at exact character boundary.
        char_end = offsets[max_tokens][0]
        return text[:char_end]

    # Symmetric trimming around the marker pair.
    marker_end_char = close_idx + len(span_close)
    text_len = max(len(text), 1)

    open_tok = int(open_idx / text_len * n)
    close_tok = int(marker_end_char / text_len * n)

    marker_tok_span = close_tok - open_tok
    if marker_tok_span >= max_tokens:
        return text[open_idx:marker_end_char]

    remaining = max_tokens - marker_tok_span
    head_budget = remaining // 2
    tail_budget = remaining - head_budget

    start_tok = max(0, open_tok - head_budget)
    end_tok = min(n, close_tok + tail_budget)

    char_start = offsets[start_tok][0]
    char_end = offsets[end_tok - 1][1] if end_tok > start_tok else char_start
    return text[char_start:char_end]


def inject_span_markers(
    text: str,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    marker_open: str = SPAN_MARKER,
    marker_close: str = SPAN_MARKER_END,
) -> str:
    """Wrap the target occurrence in markers so the LLM has an unambiguous anchor.

    If char_start and char_end are not provided, returns the text unchanged.
    """
    if char_start is None or char_end is None:
        return text
    return (
        text[:char_start]
        + marker_open
        + text[char_start:char_end]
        + marker_close
        + text[char_end:]
    )


def get_mention_context_with_marker(
    text: str,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    window: int | None = CLASSIFICATION_CONTEXT_WINDOW,
) -> str:
    """
    Return a windowed excerpt centred on [char_start, char_end], with the
    target span wrapped in >>> / <<< markers so the LLM classifies exactly
    this occurrence and ignores any unmarked repetitions nearby.

    If char_start and char_end are not provided, returns the text unchanged.
    """
    if char_start is None or char_end is None:
        return text

    if window is not None:
        excerpt_start = max(0, char_start - window)
        excerpt_end = min(len(text), char_end + window)
    else:
        excerpt_start = 0
        excerpt_end = len(text)
    excerpt = text[excerpt_start:excerpt_end]

    local_start = char_start - excerpt_start
    local_end = char_end - excerpt_start

    return inject_span_markers(
        excerpt,
        local_start,
        local_end,
    )


def build_classification_context(
    text: str,
    char_start: Optional[int],
    char_end: Optional[int],
    *,
    window: int | None,
    tokenizer=None,
    max_tokens: int = 0,
) -> str:
    """Single canonical builder for the classifier input.

    Order of operations is load-bearing:
      1. Character window around the span (full text when window=None, still marked).
      2. >>> / <<< marker injection around the span.
      3. truncate_to_token_budget() token cap (symmetric around the marker pair).

    When char_start / char_end are None, no window or markers are applied and
    the text is only token-capped from the tail.
    """
    if char_start is not None and char_end is not None:
        ctx = get_mention_context_with_marker(text, char_start, char_end, window=window)
    else:
        ctx = text
    return truncate_to_token_budget(ctx, tokenizer, max_tokens)
