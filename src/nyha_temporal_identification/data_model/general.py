from typing import Literal, Optional


class ParsedDate:
    text: str = ""
    parsed_date: Optional[str] = None
    original_start: int = -1
    start: int = -1
    original_end: int = -1
    end: int = -1
    temporal_status: Literal["current", "historical"] = "current"

    def __init__(
        self,
        text: str = "",
        parsed_date: Optional[str] = None,
        original_start: int = -1,
        start: int = -1,
        original_end: int = -1,
        end: int = -1,
        temporal_status: Literal["current", "historical"] = "current",
    ):
        self.text = text
        self.parsed_date = parsed_date
        self.original_start = original_start
        self.original_end = original_end
        self.start = start
        self.end = end
        self.temporal_status = temporal_status

    def get_json_serializable(self) -> dict:
        return {
            "text": self.text,
            "parsed_date": self.parsed_date,
            "original_start": int(self.original_start),
            "original_end": int(self.original_end),
            "start": int(self.start),
            "end": int(self.end),
            "temporal_status": self.temporal_status,
        }


class NyhaMention:
    def __init__(
        self,
        original_text: str,
        cleaned_text: str,
        classes: list[int],
        one_hot_vector: list[int],
        start: int,
        end: int,
        original_start: int,
        original_end: int,
        linked_date: Optional[ParsedDate] = None,
        char_distance: Optional[int] = None,
    ):
        assert len(classes) <= 2, "classes list must have at most 2 elements"
        assert len(one_hot_vector) == 4, "one_hot_vector list must have 4 elements"
        # At most two classes can be > 0, and they must be adjacent (e.g. [0,1,1,0] is valid, but not [1,0,1,0])
        assert sum(c > 0 for c in classes) <= 2, "At most two classes can be > 0"

        self.original_text = original_text
        self.cleaned_text = cleaned_text
        self.classes = classes
        self.one_hot_vector = one_hot_vector
        self.start = start
        self.end = end
        self.original_start = original_start
        self.original_end = original_end
        self.linked_date = linked_date
        self.char_distance = char_distance

    def get_roman_nyha_string_class(
        self,
    ) -> Literal["I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"]:
        if self.classes and len(self.classes) > 0:
            # Get all the digits in classes and separate them with "-"
            returned_nyha_class = self.get_nyha_string_class()
            # Replace the - with underscore and convert digits to roman numerals
            # (e.g. 1 -> I, 2 -> II, etc.)
            returned_nyha_class = returned_nyha_class.replace("-", "_")
            roman_numerals = ["", "I", "II", "III", "IV"]
            roman_returned_nyha_class = "-".join(
                roman_numerals[int(c)] for c in returned_nyha_class.split("_") if c
            )
            return roman_returned_nyha_class  # type: ignore
        raise ValueError("No classes found for this mention")

    def get_nyha_string_class(
        self,
    ) -> str:
        if self.classes and len(self.classes) > 0:
            # Get all the digits in classes and separate them with "-"
            returned_nyha_class = "-".join(str(c) for c in self.classes if c > 0)
            return returned_nyha_class
        raise ValueError("No classes found for this mention")

    def get_json_serializable(self) -> dict:
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "classes": [int(c) for c in self.classes],
            "one_hot_vector": [int(v) for v in self.one_hot_vector],
            "start": int(self.start),
            "end": int(self.end),
            "original_start": int(self.original_start),
            "original_end": int(self.original_end),
            "linked_date": self.linked_date.get_json_serializable()
            if self.linked_date is not None
            else None,
            "char_distance": int(self.char_distance)
            if self.char_distance is not None
            else None,
        }


class InvalidNyhaMention:
    def __init__(
        self,
        original_text: str,
        cleaned_text: str,
        start: int,
        end: int,
        original_start: int,
        original_end: int,
    ):
        self.original_text = original_text
        self.cleaned_text = cleaned_text
        self.start = start
        self.end = end
        self.original_start = original_start
        self.original_end = original_end

    def get_json_serializable(self) -> dict:
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "start": int(self.start),
            "end": int(self.end),
            "original_start": int(self.original_start),
            "original_end": int(self.original_end),
        }


class Mentions:
    def __init__(
        self,
        extracted_nyha_mentions: list[NyhaMention],
        extracted_dates: list[ParsedDate],
        extracted_invalid_nyha_mentions: list[InvalidNyhaMention] | None = None,
        pseudonymised_text: str | None = None,
    ):
        self.extracted_nyha_mentions = extracted_nyha_mentions
        self.extracted_dates = extracted_dates
        self.extracted_invalid_nyha_mentions = extracted_invalid_nyha_mentions or []
        self.pseudonymised_text = pseudonymised_text

    def get_json_serializable(self) -> dict:
        return {
            "extracted_nyha_mentions": [
                m.get_json_serializable() for m in self.extracted_nyha_mentions
            ],
            "extracted_invalid_nyha_mentions": [
                m.get_json_serializable() for m in self.extracted_invalid_nyha_mentions
            ],
            "extracted_dates": [
                d.get_json_serializable() for d in self.extracted_dates
            ],
        }
