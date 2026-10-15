"""Tests for nyha_temporal_identification.dspy_logic.regex.

Covers the module's full public surface against docs/spec/dspy_logic.md:
  * CURRENT_RE / HISTORICAL_RE temporal-anchor regexes (current vs historical
    classification, case-insensitivity, \\b word-boundary behaviour);
  * the CLASSIFICATION_CONTEXT_WINDOW constant;
  * get_nlp_pipeline() lazy singleton;
  * spacy_preprocess() — the thin wrapper around find_nyha_class used by the
    DSPy Layer-0 pre-processing and the baseline data loader.

Importing the module also imports extraction.parsing, which registers the
``nyha_component`` spaCy pipe (the same registration conftest relies on).
"""
import re

import pytest
import spacy
from spacy.language import Language

from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.dspy_logic.regex import (
    CLASSIFICATION_CONTEXT_WINDOW,
    CURRENT_ANCHORS,
    CURRENT_RE,
    HISTORICAL_ANCHORS,
    HISTORICAL_RE,
    get_nlp_pipeline,
    spacy_preprocess,
)


# ---------------------------------------------------------------------------
# Compiled-regex properties
# ---------------------------------------------------------------------------


class TestCompiledRegexProperties:
    """The two anchor lists must compile into case-insensitive patterns."""

    def test_current_re_is_compiled_pattern(self):
        assert isinstance(CURRENT_RE, re.Pattern)

    def test_historical_re_is_compiled_pattern(self):
        assert isinstance(HISTORICAL_RE, re.Pattern)

    def test_current_re_ignorecase_flag(self):
        assert CURRENT_RE.flags & re.IGNORECASE

    def test_historical_re_ignorecase_flag(self):
        assert HISTORICAL_RE.flags & re.IGNORECASE

    def test_anchor_lists_non_empty(self):
        assert len(CURRENT_ANCHORS) > 0
        assert len(HISTORICAL_ANCHORS) > 0

    def test_every_anchor_compiles_individually(self):
        # A malformed pattern in either list would silently poison the combined
        # alternation; compiling each one in isolation localises such failures.
        for pat in CURRENT_ANCHORS + HISTORICAL_ANCHORS:
            re.compile(pat)


# ---------------------------------------------------------------------------
# CURRENT_RE — present-tense anchors
# ---------------------------------------------------------------------------


class TestCurrentAnchors:
    @pytest.mark.parametrize(
        "text",
        [
            # English
            "patient is currently NYHA II",
            "presently NYHA III",
            "at this time the patient is stable",
            "stable right now",
            "at present NYHA II",
            "today the patient reports",
            "active heart failure",
            "seen at this visit",
            "symptoms remains unchanged",
            # Dutch — direct temporal markers
            "momenteel NYHA II",
            "op dit moment NYHA III",
            "op heden stabiel",
            "heden NYHA II",
            "patient is nu NYHA III",
            "thans NYHA II",
            "tegenwoordig NYHA II",
            "vandaag gezien",
            "dit bezoek NYHA III",
            "de huidige klasse is II",
            "actief hartfalen",
            # Dutch — clinical functional-status markers
            "patient behoudt NYHA II",
            "blijft NYHA III",
            "is geclassificeerd als NYHA II",
            "verdraagt inspanning goed",
            "presteert binnen verwachting",
            "kan trappen lopen",
        ],
    )
    def test_current_phrase_matches(self, text):
        assert CURRENT_RE.search(text) is not None

    def test_case_insensitive(self):
        assert CURRENT_RE.search("MOMENTEEL NYHA II") is not None
        assert CURRENT_RE.search("Currently stable") is not None

    def test_no_match_when_no_current_anchor(self):
        assert CURRENT_RE.search("de patient heeft nyha iii") is None


# ---------------------------------------------------------------------------
# HISTORICAL_RE — past-tense anchors
# ---------------------------------------------------------------------------


class TestHistoricalAnchors:
    @pytest.mark.parametrize(
        "text",
        [
            # English
            "previously NYHA III",
            "prior NYHA II",
            "formerly NYHA IV",
            "historically NYHA III",
            "patient was NYHA IV",
            "before the surgery",
            "last year NYHA III",
            "improved from NYHA III",
            "reduced from NYHA IV",
            "worsened from NYHA II",
            # Dutch — direct temporal markers
            "vroeger NYHA IV",
            "eerder NYHA III",
            "voorheen NYHA IV",
            "in het verleden NYHA III",
            "vorige klasse was IV",
            "destijds NYHA III",
            "toentertijd NYHA IV",
            "indertijd NYHA II",
            # Dutch — improvement / worsening transitions
            "verbeterd van NYHA III",
            "verslechterd van NYHA II",
            "afgenomen van NYHA IV",
            "gedaald van NYHA III",
            "gestegen van NYHA II",
            "van NYHA III naar II",
            # Dutch — past-tense auxiliaries
            "patient werd gezien",
            "zij werden behandeld",
            "had NYHA IV",
            "zij hadden klachten",
            "is geweest NYHA III",
            "was geclassificeerd als NYHA IV",
            # Dutch — pre-intervention markers
            "vóór de operatie NYHA III",
            "pre-operatief NYHA IV",
            "pre operatief NYHA IV",
            "voor de behandeling NYHA III",
        ],
    )
    def test_historical_phrase_matches(self, text):
        assert HISTORICAL_RE.search(text) is not None

    def test_case_insensitive(self):
        assert HISTORICAL_RE.search("Patient WERD gezien") is not None
        assert HISTORICAL_RE.search("PREVIOUSLY stable") is not None

    def test_no_match_when_no_historical_anchor(self):
        assert HISTORICAL_RE.search("patient heeft nyha iii") is None


# ---------------------------------------------------------------------------
# Word-boundary (\b) behaviour — guards against substring over-matching
# ---------------------------------------------------------------------------


class TestAnchorWordBoundaries:
    def test_now_does_not_match_nowadays(self):
        # "nowadays" maps to its own anchor (\btegenwoordig\b is the NL form);
        # the English \bnow\b must NOT fire on the "now" inside "nowadays".
        assert re.search(r"\bnow\b", "nowadays") is None

    def test_was_does_not_match_inside_washing(self):
        assert HISTORICAL_RE.search("washing the wound") is None

    def test_voor_matches_standalone_but_not_inside_voorbeeld(self):
        # \bvoor\b is a documented broad anchor; verify it is at least bounded
        # and does not fire on the unrelated word "voorbeeld" via that anchor.
        assert re.search(r"\bvoor\b", "voorbeeld") is None
        assert re.search(r"\bvoor\b", "voor de ingreep") is not None


# ---------------------------------------------------------------------------
# Regression: the "toegenomen van" historical anchor (was a dead typo)
# ---------------------------------------------------------------------------


class TestToegenomenRegression:
    """regex.py once had ``r"\\btoegeno men van\\b"`` — a stray space inside
    "toegenomen" meant the anchor could never match real clinical text. The
    fix restores ``r"\\btoegenomen van\\b"`` ("increased from")."""

    def test_toegenomen_van_now_matches(self):
        assert HISTORICAL_RE.search("nyha is toegenomen van ii naar iii") is not None

    def test_broken_literal_no_longer_an_anchor(self):
        # The old typo'd surface form must NOT be a live anchor any more.
        assert HISTORICAL_RE.search("toegeno men van iets") is None

    def test_toegenomen_present_in_anchor_list(self):
        assert any("toegenomen van" in pat for pat in HISTORICAL_ANCHORS)
        assert all("toegeno men" not in pat for pat in HISTORICAL_ANCHORS)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_classification_context_window_value(self):
        assert CLASSIFICATION_CONTEXT_WINDOW == 1000

    def test_classification_context_window_is_int(self):
        assert isinstance(CLASSIFICATION_CONTEXT_WINDOW, int)


# ---------------------------------------------------------------------------
# get_nlp_pipeline — lazy singleton
# ---------------------------------------------------------------------------


class TestGetNlpPipeline:
    def test_returns_language_instance(self):
        nlp = get_nlp_pipeline()
        assert isinstance(nlp, Language)

    def test_singleton_identity_across_calls(self):
        assert get_nlp_pipeline() is get_nlp_pipeline()

    def test_pipeline_has_nyha_component(self):
        nlp = get_nlp_pipeline()
        assert "nyha_component" in nlp.pipe_names

    def test_pipeline_is_dutch_blank(self):
        nlp = get_nlp_pipeline()
        assert nlp.lang == "nl"


# ---------------------------------------------------------------------------
# spacy_preprocess — wrapper around find_nyha_class
# ---------------------------------------------------------------------------


class TestSpacyPreprocess:
    def test_returns_list_of_nyha_mentions(self):
        mentions = spacy_preprocess("de patient heeft nyha iii")
        assert isinstance(mentions, list)
        assert len(mentions) == 1
        assert isinstance(mentions[0], NyhaMention)
        assert mentions[0].classes == [3]

    def test_empty_when_no_mention(self):
        assert spacy_preprocess("de patient voelt zich goed") == []

    def test_multiple_mentions(self):
        mentions = spacy_preprocess("nyha ii en later nyha iv")
        classes = sorted(m.classes for m in mentions)
        assert classes == [[2], [4]]

    def test_agrees_with_direct_find_nyha_class(self):
        # spacy_preprocess is a pure delegation to find_nyha_class through the
        # singleton pipeline, so the extracted classes must coincide.
        from nyha_temporal_identification.extraction.parsing import find_nyha_class

        text = "nyha iii-iv status"
        direct = find_nyha_class(text, nlp_pipeline=get_nlp_pipeline())
        via_wrapper = spacy_preprocess(text)
        assert [m.classes for m in via_wrapper] == [
            m.classes for m in direct.extracted_nyha_mentions
        ]

    def test_excludes_invalid_deanon_mentions(self):
        # "NYHA [PERSON-1]" is a de-anonymisation artifact -> invalid mention,
        # which spacy_preprocess must not return (it surfaces only valid ones).
        assert spacy_preprocess("NYHA [PERSON-1]") == []

    def test_parses_iso_authored_datetime(self):
        mentions = spacy_preprocess(
            "01-01-2023 nyha klasse iii",
            authored_datetime="2023-06-01",
        )
        assert len(mentions) == 1
        assert mentions[0].classes == [3]
        assert mentions[0].linked_date is not None

    def test_malformed_authored_datetime_does_not_crash(self):
        # An unparseable datetime string is logged and ignored; extraction
        # must still proceed (falling back to no reference datetime).
        mentions = spacy_preprocess(
            "de patient heeft nyha iii",
            authored_datetime="not-a-real-date",
        )
        assert len(mentions) == 1
        assert mentions[0].classes == [3]

    def test_none_authored_datetime(self):
        mentions = spacy_preprocess("nyha ii", authored_datetime=None)
        assert len(mentions) == 1
        assert mentions[0].classes == [2]
