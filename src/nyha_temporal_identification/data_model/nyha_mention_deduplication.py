from typing import List

from nyha_temporal_identification.data_model.dspy_model import NYHAMention


def _freeze_for_hash(value) -> object:
    """Convert nested containers into hashable, stable structures."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return tuple(
            sorted((key, _freeze_for_hash(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_hash(item) for item in value)
    return value


def _mention_dedupe_key(mention: NYHAMention) -> tuple:
    """Build a stable key for deduplicating mention objects."""
    return (
        mention.char_start,
        mention.char_end,
        mention.span,
        mention.get_nyha_class(),
        tuple(mention.nyha_vector),
        mention.temporality,
        mention.evidence,
        _freeze_for_hash(mention.linked_date),
        mention.char_distance,
    )


def unique_mentions(mentions: List[NYHAMention]) -> List[NYHAMention]:
    """Return mentions with duplicates removed while preserving order."""
    unique_mentions: List[NYHAMention] = []
    seen: set[tuple] = set()
    for mention in mentions:
        key = _mention_dedupe_key(mention)
        if key in seen:
            continue
        seen.add(key)
        unique_mentions.append(mention)
    return unique_mentions
