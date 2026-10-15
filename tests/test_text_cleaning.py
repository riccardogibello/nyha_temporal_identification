"""Tests for clean_text in utils/text_cleaning.py."""
import numpy as np
import pytest

from nyha_temporal_identification.constants import DEFAULT_NO_TEXT, REMOVED_SENTENCES
from nyha_temporal_identification.utils.text_cleaning import (
    LINE_BREAK_SENTINEL,
    clean_text,
)


# ---------------------------------------------------------------------------
# TestCleanTextReturnType
# ---------------------------------------------------------------------------


class TestCleanTextReturnType:
    def test_returns_tuple(self):
        result = clean_text("hello")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cleaned_is_str(self):
        cleaned, _ = clean_text("hello")
        assert isinstance(cleaned, str)

    def test_index_map_is_dict(self):
        _, index_map = clean_text("hello")
        assert isinstance(index_map, dict)

    def test_index_map_length_matches_cleaned(self):
        cleaned, index_map = clean_text("hello world")
        assert len(index_map) == len(cleaned)

    def test_index_map_keys_are_range(self):
        cleaned, index_map = clean_text("hello world")
        assert set(index_map.keys()) == set(range(len(cleaned)))

    def test_index_map_values_are_valid_indices(self):
        text = "Hello World"
        cleaned, index_map = clean_text(text)
        for val in index_map.values():
            assert 0 <= val < len(text)


# ---------------------------------------------------------------------------
# TestCleanTextInputTypes
# ---------------------------------------------------------------------------


class TestCleanTextInputTypes:
    def test_str_input(self):
        cleaned, _ = clean_text("hello")
        assert cleaned == "hello"

    def test_list_input_joined_with_space(self):
        cleaned, _ = clean_text(["hello", "world"])
        assert cleaned == "hello world"

    def test_list_single_element(self):
        cleaned, _ = clean_text(["hello"])
        assert cleaned == "hello"

    def test_numpy_1d_array(self):
        arr = np.array(["hello", "world"])
        cleaned, _ = clean_text(arr)
        assert cleaned == "hello world"

    def test_numpy_2d_array_flattened(self):
        arr = np.array([["hello", "world"]])
        cleaned, _ = clean_text(arr)
        assert cleaned == "hello world"

    def test_int_converted_via_str(self):
        cleaned, _ = clean_text(42)
        assert cleaned == "42"

    def test_none_converted_via_str(self):
        cleaned, _ = clean_text(None)
        assert cleaned == "none"


# ---------------------------------------------------------------------------
# TestCleanTextForbiddenSentences
# ---------------------------------------------------------------------------


class TestCleanTextForbiddenSentences:
    def test_first_phrase_triggers_early_exit(self):
        result = clean_text(REMOVED_SENTENCES[0])
        assert result == (DEFAULT_NO_TEXT, {})

    @pytest.mark.parametrize("phrase", REMOVED_SENTENCES)
    def test_each_phrase_triggers_early_exit(self, phrase):
        result = clean_text(phrase)
        assert result == (DEFAULT_NO_TEXT, {})

    def test_phrase_embedded_in_longer_text_triggers_early_exit(self):
        phrase = REMOVED_SENTENCES[0]
        text = "some text before " + phrase + " and some text after"
        result = clean_text(text)
        assert result == (DEFAULT_NO_TEXT, {})

    def test_no_forbidden_phrase_proceeds_normally(self):
        text = "nyha klasse ii"
        cleaned, index_map = clean_text(text)
        assert cleaned == "nyha klasse ii"
        assert len(index_map) == len(cleaned)


# ---------------------------------------------------------------------------
# TestCleanTextLowercasing
# ---------------------------------------------------------------------------


class TestCleanTextLowercasing:
    def test_uppercase_lowercased(self):
        cleaned, _ = clean_text("HELLO")
        assert cleaned == "hello"

    def test_mixed_case_lowercased(self):
        cleaned, _ = clean_text("Hello World")
        assert cleaned == "hello world"

    def test_already_lowercase_unchanged(self):
        cleaned, _ = clean_text("hello world")
        assert cleaned == "hello world"

    def test_nyha_class_lowercased(self):
        cleaned, _ = clean_text("NYHA II")
        assert cleaned == "nyha ii"


# ---------------------------------------------------------------------------
# TestCleanTextCharTranslation
# ---------------------------------------------------------------------------


class TestCleanTextCharTranslation:
    def test_newline_to_sentinel(self):
        cleaned, _ = clean_text("a\nb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_carriage_return_to_sentinel(self):
        cleaned, _ = clean_text("a\rb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_crlf_collapsed_to_single_sentinel(self):
        cleaned, _ = clean_text("a\r\nb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_tab_to_space(self):
        cleaned, _ = clean_text("a\tb")
        assert cleaned == "a b"

    def test_tab_then_space_collapses(self):
        # \t → space (first), adjacent space collapses
        cleaned, _ = clean_text("a\t b")
        assert cleaned == "a b"

    def test_space_then_tab_collapses(self):
        # space first, then \t → space; collapses
        cleaned, _ = clean_text("a \tb")
        assert cleaned == "a b"


# ---------------------------------------------------------------------------
# TestCleanTextAllowedChars
# ---------------------------------------------------------------------------


_DROPPED_CHARS = ["(", ")", "[", "]", ":", ";", "!", "?", "*", "@", "#"]


class TestCleanTextAllowedChars:
    @pytest.mark.parametrize("char", list("abcdefghijklmnopqrstuvwxyz"))
    def test_lowercase_letter_kept(self, char):
        cleaned, _ = clean_text(f"a{char}b")
        assert char in cleaned

    @pytest.mark.parametrize("char", list("0123456789"))
    def test_digit_kept(self, char):
        cleaned, _ = clean_text(f"a{char}b")
        assert char in cleaned

    @pytest.mark.parametrize("char", [".", "-", "/", " "])
    def test_allowed_special_char_kept(self, char):
        cleaned, _ = clean_text(f"a{char}b")
        assert char in cleaned

    @pytest.mark.parametrize("char", _DROPPED_CHARS)
    def test_dropped_char_absent_from_cleaned(self, char):
        cleaned, _ = clean_text(f"a{char}b")
        assert char not in cleaned

    @pytest.mark.parametrize("char", _DROPPED_CHARS)
    def test_dropped_char_has_no_index_map_entry(self, char):
        text = f"a{char}b"
        _, index_map = clean_text(text)
        # The dropped char is always at original index 1 in "a{char}b"
        assert 1 not in index_map.values()


# ---------------------------------------------------------------------------
# TestCleanTextNonAscii
# ---------------------------------------------------------------------------


class TestCleanTextNonAscii:
    def test_non_ascii_removed_by_default(self):
        # é = U+00E9, ord 233 > 127 → dropped
        cleaned, _ = clean_text("café")
        assert "é" not in cleaned
        assert cleaned == "caf"

    def test_een_reduces_to_n(self):
        # "één" = é + é + n; both é chars are non-ASCII and dropped
        cleaned, _ = clean_text("één")
        assert cleaned == "n"

    def test_non_ascii_preserved_when_normalize_false(self):
        # é is alphanumeric in Unicode, so it passes the isalnum() filter
        cleaned, _ = clean_text("café", normalize_text=False)
        assert "é" in cleaned
        assert cleaned == "café"

    def test_normalize_false_still_lowercases(self):
        cleaned, _ = clean_text("CAFÉ", normalize_text=False)
        assert cleaned == "café"

    def test_normalize_false_still_converts_newline(self):
        cleaned, _ = clean_text("a\nb", normalize_text=False)
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_normalize_false_still_collapses_whitespace(self):
        cleaned, _ = clean_text("a  b", normalize_text=False)
        assert cleaned == "a b"


# ---------------------------------------------------------------------------
# TestCleanTextWhitespaceCollapse
# ---------------------------------------------------------------------------


class TestCleanTextWhitespaceCollapse:
    def test_multiple_spaces_collapsed(self):
        cleaned, _ = clean_text("a   b")
        assert cleaned == "a b"

    def test_multiple_newlines_collapsed(self):
        cleaned, _ = clean_text("a\n\n\nb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_newline_then_space_sentinel_wins(self):
        # \n → sentinel (first), then " " collapses; sentinel survives
        cleaned, _ = clean_text("a\n b")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_space_then_newline_space_wins(self):
        # " " first → space survives; \n → sentinel but collapses
        cleaned, _ = clean_text("a \nb")
        assert cleaned == "a b"

    def test_tab_then_newline_space_wins(self):
        # \t → space (first); \n → sentinel but collapses
        cleaned, _ = clean_text("a\t\nb")
        assert cleaned == "a b"

    def test_newline_then_tab_sentinel_wins(self):
        # \n → sentinel (first); \t → space but collapses
        cleaned, _ = clean_text("a\n\tb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_crlf_single_sentinel(self):
        # \r → sentinel (first); \n → sentinel but collapses
        cleaned, _ = clean_text("a\r\nb")
        assert cleaned == "a" + LINE_BREAK_SENTINEL + "b"

    def test_only_first_whitespace_char_has_index_entry(self):
        # "a   b": only the first space (original index 1) gets a map entry
        text = "a   b"
        cleaned, index_map = clean_text(text)
        assert cleaned == "a b"
        assert index_map[1] == 1  # space in cleaned maps to first space in original
        assert 2 not in index_map.values()  # second space has no entry
        assert 3 not in index_map.values()  # third space has no entry


# ---------------------------------------------------------------------------
# TestCleanTextStripping
# ---------------------------------------------------------------------------


class TestCleanTextStripping:
    def test_leading_spaces_stripped(self):
        cleaned, _ = clean_text("  hello")
        assert cleaned == "hello"

    def test_trailing_spaces_stripped(self):
        cleaned, _ = clean_text("hello  ")
        assert cleaned == "hello"

    def test_leading_sentinel_stripped(self):
        cleaned, _ = clean_text("\nhello")
        assert cleaned == "hello"

    def test_trailing_sentinel_stripped(self):
        cleaned, _ = clean_text("hello\n")
        assert cleaned == "hello"

    def test_mixed_leading_whitespace_stripped(self):
        # " \n" → space then sentinel (space wins in collapse); leading space stripped
        cleaned, _ = clean_text(" \nhello")
        assert cleaned == "hello"

    def test_all_whitespace_returns_empty_string(self):
        cleaned, index_map = clean_text("   \n\n  ")
        assert cleaned == ""
        assert index_map == {}

    def test_only_disallowed_chars_returns_empty(self):
        cleaned, index_map = clean_text("()[]!?")
        assert cleaned == ""
        assert index_map == {}

    def test_no_leading_trailing_whitespace_unchanged(self):
        cleaned, _ = clean_text("hello world")
        assert cleaned == "hello world"

    def test_leading_space_trimmed_from_index_map(self):
        text = "  hello"
        cleaned, index_map = clean_text(text)
        assert cleaned == "hello"
        # After stripping, key 0 must map to 'h' at original index 2
        assert index_map[0] == 2


# ---------------------------------------------------------------------------
# TestCleanTextIndexMap
# ---------------------------------------------------------------------------


class TestCleanTextIndexMap:
    def test_keys_are_exactly_range(self):
        cleaned, index_map = clean_text("hello world")
        assert set(index_map.keys()) == set(range(len(cleaned)))

    def test_length_matches_cleaned(self):
        cleaned, index_map = clean_text("hello world")
        assert len(index_map) == len(cleaned)

    def test_values_strictly_increasing(self):
        cleaned, index_map = clean_text("hello world")
        vals = [index_map[i] for i in range(len(cleaned))]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_alnum_maps_to_original_lowercase_char(self):
        text = "Hello World"
        cleaned, index_map = clean_text(text)
        for i, c in enumerate(cleaned):
            if c.isalnum():
                assert text[index_map[i]].lower() == c

    def test_newline_maps_to_line_break_in_original(self):
        text = "a\nb"
        cleaned, index_map = clean_text(text)
        sentinel_pos = cleaned.index(LINE_BREAK_SENTINEL)
        assert text[index_map[sentinel_pos]] in "\n\r"

    def test_carriage_return_maps_to_line_break_in_original(self):
        text = "a\rb"
        cleaned, index_map = clean_text(text)
        sentinel_pos = cleaned.index(LINE_BREAK_SENTINEL)
        assert text[index_map[sentinel_pos]] in "\n\r"

    def test_tab_maps_to_tab_in_original(self):
        text = "a\tb"
        cleaned, index_map = clean_text(text)
        space_pos = cleaned.index(" ")
        assert text[index_map[space_pos]] == "\t"

    def test_dropped_chars_absent_from_map_values(self):
        text = "a(b)c"
        _, index_map = clean_text(text)
        # '(' at original index 1, ')' at original index 3
        assert 1 not in index_map.values()
        assert 3 not in index_map.values()

    def test_collapsed_whitespace_extras_absent_from_map_values(self):
        text = "a   b"
        _, index_map = clean_text(text)
        # Original indices 2 and 3 are the collapsed extra spaces
        assert 2 not in index_map.values()
        assert 3 not in index_map.values()

    def test_stripped_keys_start_at_zero(self):
        cleaned, index_map = clean_text("  hello  ")
        assert set(index_map.keys()) == set(range(len(cleaned)))

    def test_empty_result_has_empty_map(self):
        cleaned, index_map = clean_text("   ")
        assert cleaned == ""
        assert index_map == {}


# ---------------------------------------------------------------------------
# TestCleanTextRoundTrip
# ---------------------------------------------------------------------------


def _recover_span(original: str, index_map: dict, i: int, j: int) -> str:
    """Return the original substring corresponding to cleaned[i:j]."""
    return original[index_map[i] : index_map[j - 1] + 1]


class TestCleanTextRoundTrip:
    def test_simple_ascii(self):
        text = "NYHA klasse II"
        cleaned, index_map = clean_text(text)
        i = cleaned.index("klasse")
        j = i + len("klasse")
        assert _recover_span(text, index_map, i, j).lower() == "klasse"

    def test_embedded_newline(self):
        text = "klasse\nii"
        cleaned, index_map = clean_text(text)
        i = cleaned.index("ii")
        j = i + len("ii")
        assert _recover_span(text, index_map, i, j).lower() == "ii"

    def test_dropped_special_chars(self):
        text = "klasse (ii)"
        cleaned, index_map = clean_text(text)
        assert cleaned == "klasse ii"
        i = cleaned.index("ii")
        j = i + len("ii")
        assert _recover_span(text, index_map, i, j).lower() == "ii"

    def test_leading_whitespace_stripped(self):
        text = "  nyha ii"
        cleaned, index_map = clean_text(text)
        assert cleaned == "nyha ii"
        i = 0
        j = len("nyha")
        assert _recover_span(text, index_map, i, j).lower() == "nyha"

    def test_full_string_recovery(self):
        text = "Hello, World!"
        cleaned, index_map = clean_text(text)
        # cleaned = "hello world" (comma, '!' dropped)
        assert cleaned == "hello world"
        i, j = 0, len(cleaned)
        recovered = _recover_span(text, index_map, i, j)
        assert recovered.lower().startswith("h")
        assert recovered.lower().endswith("d")
