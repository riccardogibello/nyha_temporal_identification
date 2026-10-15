from typing import List, Literal, Optional

from pydantic import BaseModel, model_validator

from nyha_temporal_identification.constants import (
    FORMATTED_DATE_KEY,
    NBSP,
    ORIGINAL_END_KEY,
    ORIGINAL_START_KEY,
    ORIGINAL_TEXT_KEY,
    TEMPORAL_STATUS_KEY,
)
from nyha_temporal_identification.data_model.constants import (
    LABEL_TO_VECTOR,
    NYHA_MAP,
    vector_key,
)
from nyha_temporal_identification.data_model.general import NyhaMention, ParsedDate


class DspyDate(BaseModel):
    """A date found in the text and linked to a nearby NYHA mention."""

    text: str = ""
    parsed_date: Optional[str] = None
    original_start: int = -1
    original_end: int = -1
    temporal_status: Literal["current", "historical"] = "current"

    @model_validator(mode="before")
    @classmethod
    def parse_raw_fields(cls, data: dict) -> dict:
        data["text"] = data.get(ORIGINAL_TEXT_KEY, "")
        data["parsed_date"] = data.get(FORMATTED_DATE_KEY, None)
        data["original_start"] = int(data.get(ORIGINAL_START_KEY, -1))
        data["original_end"] = int(data.get(ORIGINAL_END_KEY, -1))
        data["temporal_status"] = data.get(TEMPORAL_STATUS_KEY, "current")
        return data

    @staticmethod
    def from_date(
        other_date: ParsedDate,
    ) -> "DspyDate":
        return DspyDate(
            text=other_date.text,
            parsed_date=other_date.parsed_date,
            original_start=other_date.original_start,
            original_end=other_date.original_end,
            temporal_status=other_date.temporal_status,
        )


class NYHAMention(BaseModel):
    """
    A single resolved NYHA mention with validated character indices.
    """

    char_start: int
    char_end: int
    span: str
    nyha_class: Literal[
        "I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"
    ] = "I"
    nyha_vector: List[int] = []
    temporality: Literal["current", "historical"] = "historical"
    evidence: str = ""
    linked_date: Optional[DspyDate] = None
    char_distance: Optional[int] = None
    source: Literal["llm", "failsafe"] = "llm"

    def get_nyha_class(self) -> Optional[str]:
        if self.nyha_vector and len(self.nyha_vector) == 4:
            return NYHA_MAP.get(vector_key(self.nyha_vector), None)
        return None

    @staticmethod
    def from_nyha_mention(
        other_nyha_mention: NyhaMention,
    ) -> "NYHAMention":
        temporality = (
            other_nyha_mention.linked_date.temporal_status
            if other_nyha_mention.linked_date is not None
            else "current"
        )

        return NYHAMention(
            char_start=other_nyha_mention.original_start,
            char_end=other_nyha_mention.original_end,
            span=other_nyha_mention.original_text,
            nyha_class=other_nyha_mention.get_roman_nyha_string_class(),
            nyha_vector=other_nyha_mention.one_hot_vector,
            temporality=temporality,
            evidence="",
            linked_date=DspyDate.from_date(other_nyha_mention.linked_date)
            if other_nyha_mention.linked_date
            else None,
            char_distance=other_nyha_mention.char_distance,
        )

    @model_validator(mode="before")
    @classmethod
    def parse_raw_fields(cls, data: dict) -> dict:
        data["char_start"] = int(data.get("char_start", -1))
        data["char_end"] = int(data.get("char_end", -1))
        data["span"] = data.get("span", "")
        data["nyha_vector"] = data.get("nyha_vector", [])

        valid_classes = set(
            LABEL_TO_VECTOR
        )  # {"I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"}
        existing_class = data.get("nyha_class", None)

        nyha_classes: list[int] = data.get("nyha_classes", [])
        if nyha_classes:
            sorted_classes = sorted(nyha_classes)
            key = "_".join(str(c) for c in sorted_classes)
            computed_class = NYHA_MAP.get(key, "I")

            if existing_class not in valid_classes:
                # No valid class set yet — use spaCy's extraction
                data["nyha_class"] = computed_class
            elif (
                existing_class in valid_classes
                and "-" in existing_class
                and "-" not in computed_class
            ):
                # LLM gave a range label but spaCy only found one class of the range
                # (e.g. LLM: "I-II", spaCy: [2] → "II") — keep the LLM's range label
                pass
            else:
                # Both agree or spaCy has more specific info — trust spaCy
                data["nyha_class"] = computed_class

        # Derive nyha_vector from nyha_class if not already set. Use
        # LABEL_TO_VECTOR (not an inverted NYHA_MAP) so broad-range labels resolve
        # to their collapsed class vector (e.g. "II-IV" -> [0, 1, 0, 0]).
        if not data["nyha_vector"] and "nyha_class" in data:
            class_label = data["nyha_class"]
            if class_label in LABEL_TO_VECTOR:
                data["nyha_vector"] = list(LABEL_TO_VECTOR[class_label])

        linked_date = data.get("linked_date", None)
        if linked_date and isinstance(linked_date, dict):
            linked = DspyDate(**linked_date)
            data["linked_date"] = linked
            # Override the temporality only if it is not yet set
            if "temporality" not in data:
                data["temporality"] = linked.temporal_status
        elif linked_date and isinstance(linked_date, DspyDate):
            data["linked_date"] = linked_date
            if "temporality" not in data:
                data["temporality"] = linked_date.temporal_status

        # A valid NYHA vector is a single class (e.g. [1, 0, 0, 0]) or an adjacent
        # pair (e.g. [1, 1, 0, 0] for I-II, [0, 1, 1, 0] for II-III). The single
        # invariant `vector_key(v) in NYHA_MAP` rejects empty, non-contiguous
        # (e.g. [1, 0, 1, 0]) and 3+-class (e.g. [0, 1, 1, 1]) vectors at once.
        if data["nyha_vector"]:
            if sum(data["nyha_vector"]) == 0:
                raise ValueError("nyha_vector must have at least one class indicated")
            if vector_key(data["nyha_vector"]) not in NYHA_MAP:
                raise ValueError(
                    "nyha_vector must be a single class or an adjacent pair "
                    "(a contiguous run of length 1 or 2)"
                )
        return data

    @model_validator(mode="after")
    def validate_indices(self) -> "NYHAMention":
        if self.char_start != -1 and self.char_end != -1:
            assert self.char_end > self.char_start, (
                f"char_end ({self.char_end}) must be > char_start ({self.char_start})"
            )

        assert (self.char_end - self.char_start) <= 300, (
            "Span suspiciously long — likely a hallucinated boundary"
        )
        if self.nyha_vector:
            assert len(self.nyha_vector) == 4, "nyha_vector must be length 4"
        return self

    def validate_against_text(
        self,
        deidentified_text: str,
    ) -> bool:
        if self.char_start < 0 or self.char_end > len(deidentified_text):
            return False
        # Remove from any of the two texts any occurrence of a non-breaking space
        extracted = deidentified_text[self.char_start : self.char_end]
        returned_bool = extracted.strip().lower() == self.span.strip().lower()
        if returned_bool is False:
            print(
                f"Validation failed for span '{self.span}' at indices ({self.char_start}, {self.char_end}). "
                f"Extracted text was '{extracted}'."
            )
        return returned_bool

    def overlap(self, other: "NYHAMention | NyhaMention") -> float:
        def _normalized_bounds(entity: "NYHAMention | NyhaMention") -> tuple[int, int]:
            start = getattr(entity, "char_start", None)
            if start is None:
                start = getattr(entity, "start", getattr(entity, "original_start", -1))

            end = getattr(entity, "char_end", None)
            if end is None:
                end = getattr(entity, "end", getattr(entity, "original_end", -1))

            span = getattr(
                entity,
                "span",
                getattr(entity, "cleaned_text", getattr(entity, "original_text", "")),
            )
            if isinstance(span, str) and end is not None and end >= 0:
                # NBSPs can shift span length relative to character indices.
                end -= span.count(NBSP)

            return int(start), int(end)

        self_start, self_end = _normalized_bounds(self)
        other_start, other_end = _normalized_bounds(other)

        if (
            self_start < 0
            or other_start < 0
            or self_end <= self_start
            or other_end <= other_start
        ):
            return 0.0

        overlap_start = max(self_start, other_start)
        overlap_end = min(self_end, other_end)
        overlap_len = max(0, overlap_end - overlap_start)
        union_len = max(self_end, other_end) - min(self_start, other_start)
        return overlap_len / union_len if union_len > 0 else 0.0

    @property
    def is_range_mention(self) -> bool:
        return sum(self.nyha_vector) > 1 if self.nyha_vector else False


def nyha_mention_from_json_dict(data: dict) -> NYHAMention:
    remapped_keys = {
        "original_start": "char_start",
        "original_end": "char_end",
        "original_text": "span",
        "classes": "nyha_classes",
        "one_hot": "nyha_vector",
        "linked_date": "linked_date",
        "char_distance": "char_distance",
        "temporal_status": "temporality",
    }
    # Remap the old keys to the new keys expected by NYHAMention
    new_data = {}
    for old_key, new_key in remapped_keys.items():
        if old_key in data:
            new_data[new_key] = data[old_key]

    return NYHAMention(**new_data)


class NYHAMentionRaw(BaseModel):
    """
    Raw LLM output — span text + occurrence, no character indices.
    The LLM is never asked to produce character offsets.
    """

    span: str
    nyha_class: Literal[
        "I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"
    ]
    temporality: Literal["current", "historical"]
    evidence: str
    occurrence: int = 1


class NYHANote(BaseModel):
    old_note_id: str
    deidentified_text: str
    mentions: List[NYHAMention]
    authored_datetime: Optional[str] = (
        None  # ISO format; used for relative date parsing
    )
