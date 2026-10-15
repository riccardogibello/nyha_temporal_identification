"""Tests for NYHA mention extraction: norm_token, parse_single, parse_pair_or_single,
nyha_component (via spaCy pipeline), and the full find_nyha_class pipeline."""
import datetime

import pytest

from nyha_temporal_identification.extraction.parsing import (
    _build_nyha_patterns,
    find_nyha_class,
    find_raw_nyha_spans,
    norm_token,
    parse_pair_or_single,
    parse_single,
)



# ---------------------------------------------------------------------------
# norm_token
# ---------------------------------------------------------------------------


class TestNormToken:
    def test_lowercase(self):
        assert norm_token("IV") == "iv"

    def test_l_to_i_substitution(self):
        assert norm_token("l") == "i"
        assert norm_token("ll") == "ii"
        assert norm_token("lll") == "iii"
        assert norm_token("lv") == "iv"

    def test_mixed_l_and_i(self):
        assert norm_token("Il") == "ii"
        assert norm_token("lI") == "ii"
        assert norm_token("lIl") == "iii"

    def test_collapse_repeated_hyphens(self):
        assert norm_token("i--i") == "i-i"
        assert norm_token("i---i") == "i-i"

    def test_collapse_repeated_spaces(self):
        assert norm_token("i  i") == "i i"
        assert norm_token("i   i") == "i i"

    def test_remove_all_hyphens_and_spaces(self):
        assert norm_token("i-i", remove_all=True) == "ii"
        assert norm_token("i i", remove_all=True) == "ii"
        assert norm_token("-iii-", remove_all=True) == "iii"


# ---------------------------------------------------------------------------
# parse_single
# ---------------------------------------------------------------------------


class TestParseSingle:
    @pytest.mark.parametrize(
        "token,expected_class",
        [
            # Roman numerals (clean)
            ("i", 1),
            ("ii", 2),
            ("iii", 3),
            ("iv", 4),
            # Uppercase
            ("I", 1),
            ("II", 2),
            ("III", 3),
            ("IV", 4),
            # Arabic digits
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 4),
            # English words
            ("one", 1),
            ("two", 2),
            ("three", 3),
            ("four", 4),
            # Dutch words
            ("een", 1),
            ("twee", 2),
            ("drie", 3),
            ("vier", 4),
            # OCR l-for-I artifacts
            ("l", 1),
            ("ll", 2),
            ("lll", 3),
            ("lv", 4),
            ("Il", 2),
            ("lIl", 3),
        ],
    )
    def test_valid_single(self, token, expected_class):
        result = parse_single(token)
        assert result is not None
        assert result.tolist() == [int(i == expected_class - 1) for i in range(4)]

    @pytest.mark.parametrize(
        "token",
        [
            "v",
            "0",
            "5",
            "five",
            "vijf",
            "zero",
            "null",
            "iiii",
            "",
        ],
    )
    def test_invalid_single(self, token):
        assert parse_single(token) is None


# ---------------------------------------------------------------------------
# parse_pair_or_single
# ---------------------------------------------------------------------------


class TestParsePairOrSingle:
    # -- single values are delegated to parse_single -----------------------

    def test_single_via_delegation(self):
        result = parse_pair_or_single("ii")
        assert result is not None
        assert result.tolist() == [0, 1, 0, 0]

    # -- contiguous pairs with hyphen ---------------------------------------

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("i-ii", [1, 1, 0, 0]),
            ("ii-iii", [0, 1, 1, 0]),
            ("iii-iv", [0, 0, 1, 1]),  # was previously None — now fixed
        ],
    )
    def test_contiguous_pair_hyphen(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    # -- contiguous pairs with slash (I/II and II/III allowed; III/IV excluded) --

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("i/ii", [1, 1, 0, 0]),
            ("ii/iii", [0, 1, 1, 0]),
        ],
    )
    def test_contiguous_pair_slash_allowed(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    def test_iii_iv_slash_excluded(self):
        assert parse_pair_or_single("iii/iv") is None

    def test_iii_iv_slash_with_spaces_excluded(self):
        assert parse_pair_or_single("iii / iv") is None

    # -- "X/4" scale notation: return class X only -------------------------

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("i/iv", [1, 0, 0, 0]),
            ("1/4", [1, 0, 0, 0]),
            ("i/4", [1, 0, 0, 0]),
            ("1/iv", [1, 0, 0, 0]),
            ("ii/iv", [0, 1, 0, 0]),
            ("2/4", [0, 1, 0, 0]),
            ("ii/4", [0, 1, 0, 0]),
            ("2/iv", [0, 1, 0, 0]),
            ("i / iv", [1, 0, 0, 0]),
            ("ii / iv", [0, 1, 0, 0]),
        ],
    )
    def test_x_of_4_slash_returns_single_class(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    # -- "of" and "or" separators ------------------------------------------

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ii of iii", [0, 1, 1, 0]),
            ("iii of iv", [0, 0, 1, 1]),
            ("ii or iii", [0, 1, 1, 0]),
            ("iii or iv", [0, 0, 1, 1]),
            ("i or ii", [1, 1, 0, 0]),
        ],
    )
    def test_of_or_separator(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    # -- "X of/or 4" scale notation: return class X only --------------------

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("i of iv", [1, 0, 0, 0]),
            ("1 of 4", [1, 0, 0, 0]),
            ("i of 4", [1, 0, 0, 0]),
            ("1 of iv", [1, 0, 0, 0]),
            ("ii of iv", [0, 1, 0, 0]),
            ("2 of 4", [0, 1, 0, 0]),
            ("ii of 4", [0, 1, 0, 0]),
            ("2 of iv", [0, 1, 0, 0]),
            ("i or iv", [1, 0, 0, 0]),
            ("ii or iv", [0, 1, 0, 0]),
            ("2 or 4", [0, 1, 0, 0]),
        ],
    )
    def test_x_of_4_of_or_returns_single_class(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    def test_iii_of_iv_remains_pair(self):
        # "iii of iv" is Dutch for "III or IV" (a range) — scale fix is exempted
        result = parse_pair_or_single("iii of iv")
        assert result is not None
        assert result.tolist() == [0, 0, 1, 1]

    def test_iii_or_iv_remains_pair(self):
        result = parse_pair_or_single("iii or iv")
        assert result is not None
        assert result.tolist() == [0, 0, 1, 1]

    # -- broad ranges collapse to a single class; I-III is invalid ---------

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # I-IV / II-IV: "-IV" is the scale denominator → lower class only
            ("i-iv", [1, 0, 0, 0]),
            ("ii-iv", [0, 1, 0, 0]),
        ],
    )
    def test_broad_range_collapses_to_lower_class(self, raw, expected):
        result = parse_pair_or_single(raw)
        assert result is not None
        assert result.tolist() == expected

    @pytest.mark.parametrize("raw", ["i-iii", "i/iii"])
    def test_i_iii_is_invalid(self, raw):
        # I-III is the only non-adjacent, non-scale pair → never a classed value
        assert parse_pair_or_single(raw) is None

    # -- spacing variations ------------------------------------------------

    def test_pair_with_spaces_around_hyphen(self):
        result = parse_pair_or_single("ii - iii")
        assert result is not None
        assert result.tolist() == [0, 1, 1, 0]

    def test_pair_with_spaces_around_slash_allowed(self):
        result = parse_pair_or_single("ii / iii")
        assert result is not None
        assert result.tolist() == [0, 1, 1, 0]

    # -- OCR artifacts in pairs --------------------------------------------

    def test_ocr_pair(self):
        # "ll-lll" = "ii-iii" after l→i substitution
        result = parse_pair_or_single("ll-lll")
        assert result is not None
        assert result.tolist() == [0, 1, 1, 0]

    def test_ocr_iv_pair(self):
        # "lv" = "iv" = class 4
        result = parse_pair_or_single("lll-lv")
        assert result is not None
        assert result.tolist() == [0, 0, 1, 1]

    # -- invalid inputs ----------------------------------------------------

    @pytest.mark.parametrize("raw", ["v-vi", "abc", "", "5-6"])
    def test_invalid_pair(self, raw):
        assert parse_pair_or_single(raw) is None


# ---------------------------------------------------------------------------
# nyha_component (via spaCy pipeline)
# ---------------------------------------------------------------------------


class TestNyhaComponent:
    """Tests that exercise the regex patterns inside the spaCy component."""

    def _extract(self, nlp, text):
        """Return list of (classes, one_hot) tuples for all NYHA ents in text."""
        doc = nlp(text)
        return [
            (ent._.nyha_classes, ent._.nyha_vector.tolist())
            for ent in doc.ents
            if ent.label_ == "NYHA"
        ]

    # -- pattern 2: "nyha [class_tokens] value" ----------------------------

    def test_nyha_single_arabic(self, nlp):
        result = self._extract(nlp, "nyha 2")
        assert result == [([2], [0, 1, 0, 0])]

    def test_nyha_single_roman(self, nlp):
        result = self._extract(nlp, "nyha iv")
        assert result == [([4], [0, 0, 0, 1])]

    def test_nyha_class_token(self, nlp):
        result = self._extract(nlp, "nyha class ii")
        assert result == [([2], [0, 1, 0, 0])]

    def test_nyha_klasse_token(self, nlp):
        result = self._extract(nlp, "nyha klasse iv")
        assert result == [([4], [0, 0, 0, 1])]

    def test_nyha_kl_abbrev(self, nlp):
        result = self._extract(nlp, "nyha kl. ii")
        assert result == [([2], [0, 1, 0, 0])]

    def test_nyha_hyphen_separator(self, nlp):
        result = self._extract(nlp, "nyha-iii")
        assert result == [([3], [0, 0, 1, 0])]

    def test_nyha_pair_hyphen(self, nlp):
        result = self._extract(nlp, "nyha ii-iii")
        assert result == [([2, 3], [0, 1, 1, 0])]

    def test_nyha_iii_iv_hyphen_now_captured(self, nlp):
        result = self._extract(nlp, "nyha iii-iv")
        assert result == [([3, 4], [0, 0, 1, 1])]

    def test_nyha_iii_iv_slash_excluded(self, nlp):
        result = self._extract(nlp, "nyha iii/iv")
        assert result == []

    @pytest.mark.parametrize(
        "text,expected_classes,expected_vec",
        [
            ("nyha i/iv", [1], [1, 0, 0, 0]),
            ("nyha 1/4", [1], [1, 0, 0, 0]),
            ("nyha ii/iv", [2], [0, 1, 0, 0]),
            ("nyha 2/4", [2], [0, 1, 0, 0]),
        ],
    )
    def test_nyha_x_of_4_slash_returns_single_class(
        self, nlp, text, expected_classes, expected_vec
    ):
        result = self._extract(nlp, text)
        assert result == [(expected_classes, expected_vec)]

    def test_nyha_ii_iii_slash_allowed(self, nlp):
        result = self._extract(nlp, "nyha ii/iii")
        assert result == [([2, 3], [0, 1, 1, 0])]

    def test_nyha_pair_or(self, nlp):
        result = self._extract(nlp, "nyha ii or iii")
        assert result == [([2, 3], [0, 1, 1, 0])]

    def test_nyha_pair_of(self, nlp):
        result = self._extract(nlp, "nyha iii of iv")
        assert result == [([3, 4], [0, 0, 1, 1])]

    @pytest.mark.parametrize(
        "text,expected_classes,expected_vec",
        [
            ("nyha 2 of 4", [2], [0, 1, 0, 0]),
            ("nyha 2 or 4", [2], [0, 1, 0, 0]),
            ("nyha ii of iv", [2], [0, 1, 0, 0]),
            ("nyha ii or iv", [2], [0, 1, 0, 0]),
            ("nyha 1 of 4", [1], [1, 0, 0, 0]),
        ],
    )
    def test_nyha_x_of_4_of_or_returns_single_class(
        self, nlp, text, expected_classes, expected_vec
    ):
        result = self._extract(nlp, text)
        assert result == [(expected_classes, expected_vec)]

    def test_nyha_dutch_word(self, nlp):
        result = self._extract(nlp, "nyha twee")
        assert result == [([2], [0, 1, 0, 0])]

    def test_nyha_ocr_artifact(self, nlp):
        # "ll" → "ii" after l→i
        result = self._extract(nlp, "nyha ll")
        assert result == [([2], [0, 1, 0, 0])]

    # -- pattern 1: full "new york heart association" ----------------------

    def test_full_en_name(self, nlp):
        result = self._extract(nlp, "new york heart association class ii")
        assert result == [([2], [0, 1, 0, 0])]

    def test_full_nl_name(self, nlp):
        result = self._extract(nlp, "new york hart associatie klasse iii")
        assert result == [([3], [0, 0, 1, 0])]

    def test_full_nl_name_no_assoc(self, nlp):
        # "new york hart" without association suffix is allowed (? suffix in regex)
        result = self._extract(nlp, "new york hart iii")
        assert result == [([3], [0, 0, 1, 0])]

    # -- pattern 3: reversed "value nyha" ----------------------------------

    def test_reversed_pattern(self, nlp):
        result = self._extract(nlp, "klasse iii nyha")
        assert result == [([3], [0, 0, 1, 0])]

    def test_reversed_no_class_token(self, nlp):
        result = self._extract(nlp, "iii nyha")
        assert result == [([3], [0, 0, 1, 0])]

    # -- boundary conditions: no false positives ---------------------------

    def test_no_match_nyha_alone(self, nlp):
        assert self._extract(nlp, "nyha") == []

    def test_no_match_preceded_by_letter(self, nlp):
        # "prenyha iv" — lookbehind (?<![a-z0-9]) blocks this
        assert self._extract(nlp, "prenyha iv") == []

    def test_no_match_followed_by_letter(self, nlp):
        # "nyha ivb" — lookahead (?![a-z0-9]) blocks this
        assert self._extract(nlp, "nyha ivb") == []

    def test_no_match_random_roman(self, nlp):
        # "er waren ii patienten" without "nyha" context
        assert self._extract(nlp, "er waren ii patienten zonder klachten") == []

    def test_multiple_mentions(self, nlp):
        result = self._extract(nlp, "nyha ii en later nyha iii")
        assert len(result) == 2
        assert ([2], [0, 1, 0, 0]) in result
        assert ([3], [0, 0, 1, 0]) in result

    # -- bare-space pair: Arabic digit + time word must not create false pair --

    @pytest.mark.parametrize(
        "text",
        [
            "nyha 3 4 weken geleden",
            "nyha 2 4 maanden geleden",
            "nyha 1 4 jaar geleden",
            "nyha 3 4 dagen",
        ],
    )
    def test_arabic_bare_space_temporal_phrase_no_false_pair(self, nlp, text):
        # Bare space between two Arabic digits must not form a pair when the
        # second digit is part of a temporal expression ("4 weken", etc.)
        result = self._extract(nlp, text)
        # Only the first digit should be captured as a single mention
        assert len(result) == 1
        assert result[0][0] != sorted(result[0][0] + [4])  # class 4 not present

    # -- bare-space Roman pair must still work --------------------------------

    def test_bare_space_roman_pair_still_matches(self, nlp):
        result = self._extract(nlp, "nyha ii iii")
        assert result == [([2, 3], [0, 1, 1, 0])]

    def test_arabic_hyphen_pair_still_matches(self, nlp):
        result = self._extract(nlp, "nyha 2-3")
        assert result == [([2, 3], [0, 1, 1, 0])]


# ---------------------------------------------------------------------------
# find_nyha_class (full pipeline integration)
# ---------------------------------------------------------------------------


class TestFindNyhaClass:
    def test_raises_without_pipeline(self):
        with pytest.raises(ValueError, match="nlp_pipeline"):
            find_nyha_class("nyha iv", nlp_pipeline=None)

    def test_single_mention_no_date(self, nlp):
        result = find_nyha_class("de patient heeft nyha iv", nlp_pipeline=nlp)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert m.classes == [4]
        assert m.linked_date is None

    def test_mention_with_nearby_date(self, nlp):
        text = "01-01-2023 nyha klasse iii"
        result = find_nyha_class(
            text,
            authored_datetime=datetime.datetime(2023, 6, 1),
            nlp_pipeline=nlp,
        )
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert m.classes == [3]
        assert m.linked_date is not None
        assert m.linked_date.parsed_date is not None
        assert "2023-01-01" in m.linked_date.parsed_date

    def test_no_mentions(self, nlp):
        result = find_nyha_class("de patient voelt zich goed", nlp_pipeline=nlp)
        assert result.extracted_nyha_mentions == []

    def test_original_text_positions(self, nlp):
        text = "NYHA IV"
        result = find_nyha_class(text, nlp_pipeline=nlp)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        # original_text must round-trip back to the matched fragment
        assert text[m.original_start : m.original_end].upper() == "NYHA IV"

    def test_pair_mention(self, nlp):
        result = find_nyha_class("nyha iii-iv status", nlp_pipeline=nlp)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert m.classes == [3, 4]
        assert m.one_hot_vector == [0, 0, 1, 1]

    def test_slash_pair_excluded_for_iii_iv(self, nlp):
        result = find_nyha_class("nyha iii/iv status", nlp_pipeline=nlp)
        assert result.extracted_nyha_mentions == []

    def test_multiple_mentions_link_different_dates(self, nlp):
        text = "01-01-2022 nyha ii. 01-06-2022 nyha iii."
        result = find_nyha_class(
            text,
            authored_datetime=datetime.datetime(2022, 12, 1),
            nlp_pipeline=nlp,
        )
        assert len(result.extracted_nyha_mentions) == 2
        dates = [m.linked_date.parsed_date for m in result.extracted_nyha_mentions if m.linked_date]
        assert len(dates) == 2
        assert dates[0] != dates[1]

    def test_unlinked_dates_returned(self, nlp):
        text = "01-01-2023 patient zonder nyha vermelding."
        result = find_nyha_class(
            text,
            authored_datetime=datetime.datetime(2023, 6, 1),
            nlp_pipeline=nlp,
        )
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_dates) >= 1


# ---------------------------------------------------------------------------
# Special characters: \n, \t, \r, punctuation
# ---------------------------------------------------------------------------


class TestSpecialChars:
    """Verify that whitespace and punctuation characters around NYHA mentions
    do not cause false negatives (missed mentions), false positives (spurious
    mentions), or corrupt the detected class."""

    def _classes(self, nlp, text):
        result = find_nyha_class(text, nlp_pipeline=nlp)
        return [m.classes for m in result.extracted_nyha_mentions]

    # -- trailing whitespace after the class token -------------------------

    def test_trailing_newline_after_class(self, nlp):
        assert self._classes(nlp, "NYHA III\nnext sentence") == [[3]]

    def test_trailing_tab_after_class(self, nlp):
        assert self._classes(nlp, "NYHA III\tnext word") == [[3]]

    # -- whitespace between NYHA keyword and class token -------------------

    def test_newline_between_nyha_and_class_no_spaces(self, nlp):
        # \n without flanking spaces: was silently dropped, now a space
        assert self._classes(nlp, "NYHA\nIII") == [[3]]

    def test_tab_between_nyha_and_class_no_spaces(self, nlp):
        assert self._classes(nlp, "NYHA\tIII") == [[3]]

    def test_crlf_between_nyha_and_class(self, nlp):
        assert self._classes(nlp, "NYHA\r\nIII") == [[3]]

    def test_space_newline_space_between(self, nlp):
        assert self._classes(nlp, "NYHA \n III") == [[3]]

    # -- digit on next line after class: previously a false negative -------

    def test_digit_after_newline_does_not_block_class(self, nlp):
        # "NYHA III\n2" used to clean to "nyha iii2", failing the lookahead.
        assert self._classes(nlp, "NYHA III\n2. volgende punt") == [[3]]

    def test_arabic_class_digit_after_newline(self, nlp):
        assert self._classes(nlp, "NYHA 2\n3 andere waarde") == [[2]]

    def test_iv_followed_by_digit_after_newline(self, nlp):
        assert self._classes(nlp, "NYHA IV\n1 meer tekst") == [[4]]

    # -- Roman numeral / letter on next line: previously wrong class -------

    def test_lowercase_i_after_newline_does_not_inflate_class(self, nlp):
        # "NYHA II\ni." used to clean to "nyha iii.", returning class 3 instead of 2.
        assert self._classes(nlp, "NYHA II\ni. eerste punt") == [[2]]

    def test_lowercase_l_after_newline_does_not_corrupt_class(self, nlp):
        # "NYHA III\nl" used to clean to "nyha iiil", failing the lookahead.
        assert self._classes(nlp, "NYHA III\nl. list item") == [[3]]

    def test_uppercase_roman_after_newline(self, nlp):
        # "NYHA III\nI. eerste punt" → cleaned was "nyha iiii.", no match.
        assert self._classes(nlp, "NYHA III\nI. eerste punt") == [[3]]

    def test_uppercase_roman_ii_after_newline(self, nlp):
        assert self._classes(nlp, "NYHA II\nII. tweede punt") == [[2]]

    # -- date immediately before NYHA separated only by \n ----------------

    def test_date_and_nyha_separated_by_newline(self, nlp):
        # "01-01-2023\nNYHA III" used to clean to "01-01-2023nyha iii",
        # where the digit "3" blocked the lookbehind on "nyha".
        result = find_nyha_class(
            "01-01-2023\nNYHA III",
            authored_datetime=datetime.datetime(2023, 6, 1),
            nlp_pipeline=nlp,
        )
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [3]

    def test_date_nyha_newline_link_preserved(self, nlp):
        result = find_nyha_class(
            "01-01-2023\nNYHA III",
            authored_datetime=datetime.datetime(2023, 6, 1),
            nlp_pipeline=nlp,
        )
        m = result.extracted_nyha_mentions[0]
        assert m.linked_date is not None
        assert m.linked_date.parsed_date is not None
        assert "2023-01-01" in m.linked_date.parsed_date

    # -- punctuation around NYHA mentions ----------------------------------

    def test_colon_after_nyha(self, nlp):
        assert self._classes(nlp, "NYHA: III") == [[3]]

    def test_parens_around_class(self, nlp):
        assert self._classes(nlp, "NYHA (III)") == [[3]]

    # -- newline inside full "New York Heart Association" phrase -----------

    def test_newline_inside_full_name(self, nlp):
        assert self._classes(nlp, "new york\nheart association III") == [[3]]


# ---------------------------------------------------------------------------
# `+` suffix: NYHA class-value modifier (e.g. "NYHA I+", "NYHA I+/4")
# ---------------------------------------------------------------------------


class TestPlusSuffix:
    """The `+` sign may appear directly after a class value (no space) as a
    clinical grade modifier meaning "at the upper end of this class".
    It must NOT be consumed as a separator or value token, and the span must
    still be classified correctly.

    `+` is preserved in the cleaned text (it is in ALLOWED_CHARS) so it acts
    as a natural word-boundary between the value token and any following `/N`
    scale-denominator.  This unambiguously resolves `NYHA III+/IV` as class 3
    (the `+` prevents the regex from forming the `III/IV` pair that would
    otherwise be excluded as ambiguous).
    """

    def _classes(self, nlp, text):
        result = find_nyha_class(text, nlp_pipeline=nlp)
        return [m.classes for m in result.extracted_nyha_mentions]

    # -- originally verified cases (regression anchors) --------------------

    def test_case1_nyha_range_with_slash_denominator(self, nlp):
        # "I-II/IV": regex captures "I-II"; /IV denominator is ignored (correct)
        assert self._classes(
            nlp,
            "hartfalen (NYHA I-II/IV) bij ischemische cardiomyopathie",
        ) == [[1, 2]]

    def test_case2_nyha_in_bullet_after_newline(self, nlp):
        # Newline + many spaces + "- NYHA II, CCS I": comma stripped; CCS I not absorbed
        assert self._classes(
            nlp,
            "hier vooralsnog vanaf.\n                                 - NYHA II, CCS I",
        ) == [[2]]

    # -- bare `+` suffix (no denominator) ----------------------------------

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("NYHA 1+", [[1]]),
            ("NYHA I+", [[1]]),
            ("NYHA II+", [[2]]),
            ("NYHA III+", [[3]]),
            ("NYHA IV+", [[4]]),
        ],
    )
    def test_plus_suffix_bare(self, nlp, text, expected):
        assert self._classes(nlp, text) == expected

    # -- `+` followed by /4 or /IV scale denominator -----------------------

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("NYHA I+/4", [[1]]),
            ("NYHA I+/IV", [[1]]),
            ("NYHA II+/4", [[2]]),
            ("NYHA II+/IV", [[2]]),
            ("NYHA III+/4", [[3]]),   # without `+`, "III/4" was excluded as ambiguous
            ("NYHA III+/IV", [[3]]),  # without `+`, "III/IV" was excluded as ambiguous
        ],
    )
    def test_plus_slash_denominator(self, nlp, text, expected):
        assert self._classes(nlp, text) == expected

    # -- `+` on last value of a range pair ---------------------------------

    def test_plus_on_last_value_of_range(self, nlp):
        # "NYHA I-II+" → regex sees "i-ii" (+ is boundary); classes 1 and 2
        assert self._classes(nlp, "NYHA I-II+") == [[1, 2]]

    # -- guard: `+` with a space before it is NOT a value modifier ---------

    def test_plus_with_leading_space_does_not_create_false_match(self, nlp):
        # "NYHA +3": `+` is preserved but `value_single` requires [il|iv|1-4|...]
        # at the start of the group → no match
        assert self._classes(nlp, "NYHA +3") == []

    # -- guard: `III/IV` without `+` remains excluded ----------------------

    def test_iii_iv_slash_without_plus_still_excluded(self, nlp):
        assert self._classes(nlp, "NYHA III/IV") == []


# ---------------------------------------------------------------------------
# De-anonymization placeholder detection → invalid mentions
# ---------------------------------------------------------------------------


class TestInvalidNyhaMentions:
    """Tests for de-anonymization-corrupted NYHA mentions.

    The de-identification pipeline sometimes mistakes a Roman numeral class
    (I, II, III, IV) for a name initial and replaces it with a placeholder
    like [PERSON-1] or [DOCTOR-2], producing e.g. "NYHA [PERSON-1]" instead
    of "NYHA I".  find_nyha_class must detect these bare-placeholder patterns
    as invalid mentions.  Mentions that still carry a numeral (e.g.
    "NYHA II [DOCTOR-1]") are valid — only the trailing placeholder is noise.
    """

    def _run(self, nlp, text):
        return find_nyha_class(text, nlp_pipeline=nlp)

    # -- bare placeholder (no numeral) → invalid ----------------------------

    def test_person_placeholder_no_numeral_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA [PERSON-1]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_doctor_placeholder_no_numeral_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA [DOCTOR-2]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_organization_placeholder_no_numeral_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA [ORGANIZATION-3]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_placeholder_with_underscore_type_name_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA [DOCTOR_NAME-1]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_class_word_then_placeholder_is_invalid(self, nlp):
        # "NYHA class [PERSON-1]": class-word present but numeral replaced
        result = self._run(nlp, "NYHA class [PERSON-1]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_klasse_word_then_placeholder_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA klasse [PERSON-1]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    def test_higher_placeholder_index_is_invalid(self, nlp):
        result = self._run(nlp, "NYHA [PERSON-12]")
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 1

    # -- numeral present → valid (placeholder is unrelated noise) -----------

    def test_numeral_i_with_trailing_placeholder_is_valid(self, nlp):
        result = self._run(nlp, "NYHA I [PERSON-1]")
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [1]
        assert result.extracted_invalid_nyha_mentions == []

    def test_numeral_ii_with_trailing_placeholder_is_valid(self, nlp):
        result = self._run(nlp, "NYHA II [DOCTOR-1]")
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [2]
        assert result.extracted_invalid_nyha_mentions == []

    def test_numeral_iii_with_trailing_placeholder_is_valid(self, nlp):
        result = self._run(nlp, "NYHA III [ORGANIZATION-3]")
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [3]
        assert result.extracted_invalid_nyha_mentions == []

    # -- field correctness --------------------------------------------------

    def test_invalid_mention_original_text_roundtrip(self, nlp):
        text = "NYHA [PERSON-1]"
        result = self._run(nlp, text)
        m = result.extracted_invalid_nyha_mentions[0]
        assert text[m.original_start : m.original_end] == m.original_text

    def test_invalid_mention_original_text_starts_with_nyha(self, nlp):
        result = self._run(nlp, "NYHA [PERSON-1]")
        m = result.extracted_invalid_nyha_mentions[0]
        assert m.original_text.upper().startswith("NYHA")

    def test_invalid_mention_cleaned_text_lowercased(self, nlp):
        result = self._run(nlp, "NYHA [DOCTOR-1]")
        m = result.extracted_invalid_nyha_mentions[0]
        assert "nyha" in m.cleaned_text

    def test_invalid_mention_positions_non_negative(self, nlp):
        result = self._run(nlp, "NYHA [PERSON-1]")
        m = result.extracted_invalid_nyha_mentions[0]
        assert m.start >= 0
        assert m.end > m.start
        assert m.original_start >= 0
        assert m.original_end > m.original_start

    def test_invalid_mention_json_serializable(self, nlp):
        result = self._run(nlp, "NYHA [PERSON-1]")
        d = result.extracted_invalid_nyha_mentions[0].get_json_serializable()
        for key in ("original_text", "cleaned_text", "start", "end", "original_start", "original_end"):
            assert key in d

    # -- placeholder far from NYHA does not trigger invalidation ------------

    def test_placeholder_several_words_away_is_valid(self, nlp):
        result = self._run(nlp, "NYHA III. De behandelend [DOCTOR-1] is aanwezig.")
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [3]
        assert result.extracted_invalid_nyha_mentions == []

    def test_placeholder_after_sentence_boundary_is_valid(self, nlp):
        result = self._run(nlp, "NYHA IV.\n[PERSON-1] werd gezien op de poli.")
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_invalid_nyha_mentions == []

    # -- mixed valid and invalid mentions ------------------------------------

    def test_invalid_and_valid_mention_coexist(self, nlp):
        # First mention had numeral replaced; second is clean.
        text = "NYHA [PERSON-1] en later NYHA III"
        result = self._run(nlp, text)
        assert len(result.extracted_invalid_nyha_mentions) == 1
        assert len(result.extracted_nyha_mentions) == 1
        assert result.extracted_nyha_mentions[0].classes == [3]

    def test_multiple_invalid_mentions(self, nlp):
        text = "NYHA [PERSON-1] en NYHA [DOCTOR-2]"
        result = self._run(nlp, text)
        assert result.extracted_nyha_mentions == []
        assert len(result.extracted_invalid_nyha_mentions) == 2

    def test_multiple_valid_unaffected_by_unrelated_placeholder(self, nlp):
        text = "NYHA II en NYHA III. [PERSON-1] is de patiënt."
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 2
        assert result.extracted_invalid_nyha_mentions == []


# ---------------------------------------------------------------------------
# Sentinel text: original-position mapping
# ---------------------------------------------------------------------------


class TestSentinelTextPositions:
    """Verify that find_nyha_class correctly maps cleaned-text positions back to
    original-text positions when the original text contains newlines, tabs, and
    other characters transformed or collapsed by clean_text."""

    def _run(self, nlp, text):
        return find_nyha_class(text, nlp_pipeline=nlp)

    def test_roundtrip_plain(self, nlp):
        text = "NYHA III"
        m = self._run(nlp, text).extracted_nyha_mentions[0]
        assert "III" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_with_leading_newline(self, nlp):
        text = "\nNYHA III"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert "III" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_with_prefix_text_and_newline(self, nlp):
        text = "Diagnose:\nNYHA IV"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert "IV" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_with_tab(self, nlp):
        text = "NYHA\tIII"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert "III" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_multi_newline_prefix(self, nlp):
        text = "\n\n\nNYHA II"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert "II" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_newline_inside_phrase(self, nlp):
        # The sentinel covers both NYHA and its class even when split by a newline
        text = "NYHA\nIII"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        fragment = text[m.original_start : m.original_end].upper()
        assert "NYHA" in fragment or "III" in fragment

    def test_roundtrip_mention_at_position_zero(self, nlp):
        text = "NYHA IV status"
        m = self._run(nlp, text).extracted_nyha_mentions[0]
        assert m.original_start == 0
        assert "IV" in text[m.original_start : m.original_end].upper()

    def test_roundtrip_mention_at_end_of_text(self, nlp):
        text = "status: NYHA II"
        result = self._run(nlp, text)
        assert len(result.extracted_nyha_mentions) == 1
        m = result.extracted_nyha_mentions[0]
        assert "II" in text[m.original_start : m.original_end].upper()

    def test_char_distance_none_when_no_date(self, nlp):
        result = self._run(nlp, "NYHA II")
        assert result.extracted_nyha_mentions[0].char_distance is None

    def test_char_distance_non_negative_when_date_linked(self, nlp):
        result = find_nyha_class(
            "01-01-2023 NYHA III",
            authored_datetime=datetime.datetime(2023, 6, 1),
            nlp_pipeline=nlp,
        )
        m = result.extracted_nyha_mentions[0]
        assert m.char_distance is not None
        assert m.char_distance >= 0

    def test_cleaned_text_field_lowercased(self, nlp):
        result = self._run(nlp, "NYHA IV")
        m = result.extracted_nyha_mentions[0]
        assert m.cleaned_text == m.cleaned_text.lower()

    def test_one_hot_vector_and_classes_consistent(self, nlp):
        result = self._run(nlp, "NYHA III")
        m = result.extracted_nyha_mentions[0]
        # one_hot position 2 (0-indexed) should be 1 for class 3
        assert m.one_hot_vector[2] == 1
        assert 3 in m.classes


# ---------------------------------------------------------------------------
# Sentinel text: cross-line pair blocking
# ---------------------------------------------------------------------------


class TestSentinelCrossLinePairBlocking:
    r"""Verify that value pairs cannot span a line break (sentinel \x00 is not
    \s in Python re), while the NYHA phrase itself can still cross line breaks
    via the sep pattern which explicitly allows \x00."""

    def _classes(self, nlp, text):
        return [m.classes for m in find_nyha_class(text, nlp_pipeline=nlp).extracted_nyha_mentions]

    def test_arabic_digit_on_next_line_not_paired(self, nlp):
        # "NYHA III\n2 weken" → sentinel "nyha iii\x002 weken"
        # \x00 is not \s, so no pair is formed; result is class 3 only
        assert self._classes(nlp, "NYHA III\n2 weken geleden") == [[3]]

    def test_arabic_pair_class_1_across_line_not_formed(self, nlp):
        # "NYHA 1\n4 maanden" → class 1 only
        assert self._classes(nlp, "NYHA 1\n4 maanden geleden") == [[1]]

    def test_roman_numeral_on_next_line_not_absorbed_into_pair(self, nlp):
        # "NYHA I\nIV" — IV is on the next line and must not form a class-[1,4] pair
        assert self._classes(nlp, "NYHA I\nIV volgende") == [[1]]

    def test_roman_pair_across_line_not_formed(self, nlp):
        # "NYHA II\nIII scores" — III is on the next line; result is class 2 only
        assert self._classes(nlp, "NYHA II\nIII scores") == [[2]]

    def test_phrase_sep_allows_cross_line_full_name(self, nlp):
        # sep (not \s) allows \x00, so "new york\nheart association" still matches
        assert self._classes(nlp, "new york\nheart association iii") == [[3]]

    def test_multi_newline_phrase_still_matches(self, nlp):
        # NYHA keyword, Dutch class-word, and value each separated by \n
        assert self._classes(nlp, "NYHA\nklasse\niii") == [[3]]

    def test_strong_sep_hyphen_on_same_line_still_pairs(self, nlp):
        # A hyphen pair on the same line must still work
        assert self._classes(nlp, "NYHA ii-iii") == [[2, 3]]

    def test_strong_sep_slash_on_same_line_still_pairs(self, nlp):
        assert self._classes(nlp, "NYHA ii/iii") == [[2, 3]]


# ---------------------------------------------------------------------------
# Known-bug fixes and documented limitations
# ---------------------------------------------------------------------------


class TestKnownBugsFixed:
    """Bug 1 (lv FP) and Bug 4 (reversed ranges) are fixed.
    Bugs 5 and 7 remain as xfail documentation of known limitations."""

    def _classes(self, nlp, text):
        return [m.classes for m in find_nyha_class(text, nlp_pipeline=nlp).extracted_nyha_mentions]

    # -- Bug 1: lv false positive fixed via negative lookahead --------------

    def test_lv_before_word_no_false_positive(self, nlp):
        assert self._classes(nlp, "NYHA lv status") == []

    def test_lv_ejectie_no_false_positive(self, nlp):
        assert self._classes(nlp, "NYHA lv ejectie") == []

    def test_lv_functie_no_false_positive(self, nlp):
        assert self._classes(nlp, "NYHA lv functie") == []

    def test_lv_dysfunctie_no_false_positive(self, nlp):
        assert self._classes(nlp, "NYHA lv dysfunctie") == []

    def test_lv_at_end_of_text_still_matches(self, nlp):
        # OCR case: "IV" misread as "lv"; no word follows → lookahead passes
        assert self._classes(nlp, "NYHA lv") == [[4]]

    def test_lv_before_punctuation_still_matches(self, nlp):
        # "NYHA lv." — period is not a letter; OCR case preserved
        assert self._classes(nlp, "NYHA lv.") == [[4]]

    def test_ocr_ll_regression_class2(self, nlp):
        # [il]{1,3} still captures "ll" → norm "ii" → class 2
        assert self._classes(nlp, "NYHA ll") == [[2]]

    def test_ocr_lll_regression_class3(self, nlp):
        assert self._classes(nlp, "NYHA lll") == [[3]]

    def test_lv_pair_with_word_no_false_pair(self, nlp):
        # "NYHA ii lv ejectie" — lv guarded in roman_single too; class 2 only
        assert self._classes(nlp, "NYHA ii lv ejectie") == [[2]]

    # -- Bug 4: reversed ranges now return None / no mention ---------------

    def test_reversed_iv_i_parse_returns_none(self):
        assert parse_pair_or_single("iv-i") is None

    def test_reversed_4_1_parse_returns_none(self):
        assert parse_pair_or_single("4-1") is None

    def test_reversed_iii_i_parse_returns_none(self):
        assert parse_pair_or_single("iii-i") is None

    def test_reversed_iii_ii_parse_returns_none(self):
        assert parse_pair_or_single("iii-ii") is None

    def test_reversed_iv_i_no_mention(self, nlp):
        assert self._classes(nlp, "NYHA IV-I") == []

    def test_reversed_4_1_no_mention(self, nlp):
        assert self._classes(nlp, "NYHA 4-1") == []

    def test_reversed_3_1_no_mention(self, nlp):
        assert self._classes(nlp, "NYHA 3-1") == []

    def test_forward_range_ii_iii_regression(self, nlp):
        # Forward ranges must be unaffected by the reversed-range guard
        assert self._classes(nlp, "NYHA ii-iii") == [[2, 3]]

    def test_forward_range_iii_iv_regression(self, nlp):
        assert self._classes(nlp, "NYHA iii-iv") == [[3, 4]]

    def test_forward_range_i_ii_regression(self, nlp):
        assert self._classes(nlp, "NYHA i-ii") == [[1, 2]]

    def test_equal_pair_is_single_class(self):
        # "ii-ii" collapses to single class 2 (va == vb, not reversed)
        result = parse_pair_or_single("ii-ii")
        assert result is not None
        assert result.tolist() == [0, 1, 0, 0]

    # -- Bug 5: three-value chain silently dropped (known limitation) -------

    def test_three_value_chain_extracts_at_least_partial(self, nlp):
        # Regex captures "i-ii" (first pair); alignment_mode="expand" in char_span
        # snaps the end to the token boundary of the full "i-ii-iii" token.
        # Classes [1,2] are extracted from the captured group, not the span text.
        result = self._classes(nlp, "NYHA i-ii-iii")
        assert result == [[1, 2]]

    # -- Bug 7: Dutch één accent stripped (known limitation) ---------------

    @pytest.mark.xfail(
        reason=(
            "clean_text strips characters with ord > 127, so 'é' in 'één' is "
            "removed, producing 'nyha n' with no valid class token. "
            "Use 'NYHA een' (unaccented) as the workaround."
        ),
        strict=True,
    )
    def test_dutch_een_accented_matches_class1(self, nlp):
        result = self._classes(nlp, "NYHA één")
        assert result == [[1]]


# ---------------------------------------------------------------------------
# _build_nyha_patterns
# ---------------------------------------------------------------------------


class TestBuildNyhaPatterns:
    def test_returns_three_patterns(self):
        patterns = _build_nyha_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 3

    def test_all_patterns_are_strings(self):
        for pat in _build_nyha_patterns():
            assert isinstance(pat, str)


# ---------------------------------------------------------------------------
# find_raw_nyha_spans
# ---------------------------------------------------------------------------


class TestFindRawNyhaSpans:
    def test_basic_nyha_iii(self):
        spans = find_raw_nyha_spans("NYHA III")
        assert len(spans) == 1
        s, e = spans[0]
        assert s == 0
        assert e == 8

    def test_case_insensitive(self):
        spans = find_raw_nyha_spans("nyha ii")
        assert len(spans) == 1

    def test_offsets_match_substring(self):
        text = "Patient NYHA II status."
        spans = find_raw_nyha_spans(text)
        assert len(spans) >= 1
        s, e = spans[0]
        assert text[s:e].upper().startswith("NYHA")

    def test_merges_overlapping_intervals(self):
        # "NYHA III NYHA": pattern 2 matches "NYHA III" and pattern 3
        # matches "III NYHA" — the two intervals overlap and must be merged.
        text = "NYHA III NYHA"
        spans = find_raw_nyha_spans(text)
        assert len(spans) == 1
        s, e = spans[0]
        assert s == 0
        assert e == len(text)

    def test_invalid_value_still_returned(self):
        # "NYHA IV-I" is a reversed range that parse_pair_or_single rejects,
        # but find_raw_nyha_spans must still return the interval.
        text = "NYHA IV-I"
        spans = find_raw_nyha_spans(text)
        assert len(spans) == 1

    def test_raw_newline_can_cross_line(self):
        # In raw text \s matches \n, so a pair split across lines is captured
        # (intentional over-match relative to nyha_component).
        text = "NYHA II\nIII rest of note"
        spans = find_raw_nyha_spans(text)
        assert len(spans) >= 1
        s, e = spans[0]
        # The match must span the newline (start before and end after position 7)
        assert s == 0
        assert e > 7

    def test_no_match_returns_empty(self):
        assert find_raw_nyha_spans("No heart failure mentioned.") == []

    def test_two_separate_mentions(self):
        text = "NYHA II at admission and NYHA III at discharge."
        spans = find_raw_nyha_spans(text)
        assert len(spans) == 2
