"""Tests for utils/pseudonymisation.py."""
import logging
import random
import re

import pytest

from nyha_temporal_identification.constants import DATE_PATTERNS
from nyha_temporal_identification.utils.pseudonymisation import (
    CATEGORY_ALIASES,
    DEANON_TOKEN_RE,
    PASSTHROUGH_CATEGORIES,
    PersonNameGenerator,
    InstitutionNameGenerator,
    PlaceNameGenerator,
    PseudonymisationResult,
    Pseudonymiser,
    Replacement,
    _DUTCH_NUMBER_WORDS,
    make_note_rng,
    random_date,
    random_email,
    random_id,
    random_number_word,
    random_phone,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pseudo_data")
    (d / "dutch_first_names.csv").write_text(
        "name,gender,weight\nAnna,f,100\nJan,m,100\nPiet,m,50\nSanne,f,80\n"
    )
    (d / "dutch_surnames.csv").write_text(
        "name,weight\nJansen,100\nDe Vries,80\nSmit,90\n"
    )
    (d / "dutch_places.csv").write_text(
        "name,weight\nAmsterdam,900000\nRotterdam,650000\nDen Haag,550000\n"
    )
    (d / "dutch_hospitals.csv").write_text(
        "name\nCatharina Ziekenhuis\nErasmus MC\nAMC\n"
    )
    return str(d)


@pytest.fixture(scope="module")
def pseudo(data_dir):
    return Pseudonymiser(data_dir=data_dir)


# ---------------------------------------------------------------------------
# 1. DEANON_TOKEN_RE
# ---------------------------------------------------------------------------

class TestDeanonTokenRe:
    def test_person_with_counter(self):
        m = DEANON_TOKEN_RE.fullmatch("[PERSON-1]")
        assert m is not None
        assert m.group(1) == "PERSON"
        assert m.group(2) == "1"

    def test_doctor_with_counter(self):
        m = DEANON_TOKEN_RE.fullmatch("[DOCTOR-2]")
        assert m is not None
        assert m.group(1) == "DOCTOR"
        assert m.group(2) == "2"

    def test_persoon_with_large_counter(self):
        m = DEANON_TOKEN_RE.fullmatch("[PERSOON-10]")
        assert m is not None
        assert m.group(1) == "PERSOON"
        assert m.group(2) == "10"

    def test_counterless_patient(self):
        m = DEANON_TOKEN_RE.fullmatch("[PATIENT]")
        assert m is not None
        assert m.group(1) == "PATIENT"
        assert m.group(2) is None

    def test_underscore_in_label(self):
        m = DEANON_TOKEN_RE.fullmatch("[PATIENT_NAME-3]")
        assert m is not None
        assert m.group(1) == "PATIENT_NAME"

    def test_rejects_lowercase(self):
        assert DEANON_TOKEN_RE.fullmatch("[person-1]") is None

    def test_rejects_stray_brackets(self):
        assert DEANON_TOKEN_RE.fullmatch("[ ]") is None
        assert DEANON_TOKEN_RE.fullmatch("[]") is None

    def test_finds_tokens_in_text(self):
        text = "NYHA [PERSON-1] klasse en ook [DATE-2] geleden"
        matches = DEANON_TOKEN_RE.findall(text)
        # findall returns tuples (group1, group2)
        labels = [m[0] for m in matches]
        assert "PERSON" in labels
        assert "DATE" in labels


# ---------------------------------------------------------------------------
# 2. Category alias resolution
# ---------------------------------------------------------------------------

class TestCategoryResolution:
    @pytest.mark.parametrize("raw,expected", [
        ("PERSON", "person"),
        ("PERSOON", "person"),
        ("PATIENT", "person"),
        ("DOCTOR", "person"),
        ("HOSPITAL", "institution"),
        ("ZIEKENHUIS", "institution"),
        ("INSTELLING", "institution"),
        ("LOCATION", "place"),
        ("LOCATIE", "place"),
        ("DATE", "date"),
        ("DATUM", "date"),
        ("AGE", "age"),
        ("LEEFTIJD", "age"),
        ("PHONE", "phone"),
        ("TELEFOONNUMMER", "phone"),
        ("EMAIL", "email"),
        ("EMAILADRES", "email"),
        ("ID", "id"),
        ("BSN", "id"),
        ("URL", "url"),
    ])
    def test_known_aliases(self, raw, expected):
        assert CATEGORY_ALIASES[raw] == expected

    def test_zorginstelling_maps_to_institution(self):
        assert CATEGORY_ALIASES["ZORGINSTELLING"] == "institution"

    def test_nummerwoord_maps_to_nummerwoord(self):
        assert CATEGORY_ALIASES["NUMMERWOORD"] == "nummerwoord"

    def test_nummerwoord_generates_dutch_number_word(self, data_dir):
        p = Pseudonymiser(data_dir=data_dir)
        rng = random.Random(0)
        result = p.pseudonymise("[NUMMERWOORD-1]", rng)
        assert result.new_text in _DUTCH_NUMBER_WORDS

    def test_unknown_label_emits_warning_once(self, data_dir, caplog):
        fresh = Pseudonymiser(data_dir=data_dir)
        rng = random.Random(1)
        logger_name = "nyha_temporal_identification.utils.pseudonymisation"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result1 = fresh.pseudonymise("[ONBEKENDLABEL-1]", rng)
            result2 = fresh.pseudonymise("[ONBEKENDLABEL-1]", rng)
        # Warning must have been emitted at least once
        assert any("ONBEKENDLABEL" in r.getMessage() for r in caplog.records)
        # New text must not contain the original bracket token
        assert "[ONBEKENDLABEL-1]" not in result1.new_text
        assert "[ONBEKENDLABEL-1]" not in result2.new_text

    def test_unknown_label_warns_only_once(self, data_dir, caplog):
        # Use a fresh Pseudonymiser so _warned_categories is empty and state is clean.
        fresh = Pseudonymiser(data_dir=data_dir)
        rng = random.Random(99)
        logger_name = "nyha_temporal_identification.utils.pseudonymisation"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            for _ in range(5):
                fresh.pseudonymise("[UNIQUELABEL_ONCE-1]", rng)
        warning_count = sum(
            1 for r in caplog.records if "UNIQUELABEL_ONCE" in r.getMessage()
        )
        assert warning_count == 1


# ---------------------------------------------------------------------------
# 3. Consistency within a note
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_same_token_gets_same_replacement(self, pseudo):
        rng = make_note_rng(42, "note_a")
        text = "Patiënt [PERSON-1] en later [PERSON-1] gezien."
        result = pseudo.pseudonymise(text, rng)
        # Both occurrences of [PERSON-1] must be replaced with the same value
        r1, r2 = result.replacements
        assert r1.replacement == r2.replacement

    def test_different_indices_get_different_replacements(self, pseudo):
        rng = make_note_rng(42, "note_b")
        text = "[PERSON-1] en [PERSON-2]"
        result = pseudo.pseudonymise(text, rng)
        r1, r2 = result.replacements
        assert r1.replacement != r2.replacement

    def test_different_raw_categories_same_index_are_keyed_independently(self, pseudo):
        # [PERSON-1] and [DOCTOR-1] are different cache keys even though both map
        # to the same canonical generator.  Each draws independently from the rng.
        rng = make_note_rng(42, "note_c")
        text = "[PERSON-1] werkt in [DOCTOR-1]"
        result = pseudo.pseudonymise(text, rng)
        assert ("PERSON", "1") in result.mapping
        assert ("DOCTOR", "1") in result.mapping
        # The two entries are independent (distinct cache keys), not copied from each other.
        # With a sufficiently large name pool they will almost always differ; confirm
        # the mapping has two separate entries regardless of accidental value equality.
        assert len(result.replacements) == 2


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_and_note_id_gives_same_output(self, pseudo):
        text = "Dr. [DOCTOR-1] werkt in [HOSPITAL-1]."
        rng_a = make_note_rng(123, "note_42")
        rng_b = make_note_rng(123, "note_42")
        result_a = pseudo.pseudonymise(text, rng_a)
        result_b = pseudo.pseudonymise(text, rng_b)
        assert result_a.new_text == result_b.new_text

    def test_different_note_ids_likely_different_output(self, pseudo):
        text = "[PERSON-1] is de patiënt."
        outputs = set()
        for i in range(10):
            rng = make_note_rng(42, f"note_{i}")
            outputs.add(pseudo.pseudonymise(text, rng).new_text)
        # With 10 different note IDs and a name pool > 1, expect > 1 distinct result
        assert len(outputs) > 1

    def test_make_note_rng_stable_across_calls(self):
        r1 = make_note_rng(7, "abc")
        r2 = make_note_rng(7, "abc")
        vals1 = [r1.random() for _ in range(5)]
        vals2 = [r2.random() for _ in range(5)]
        assert vals1 == vals2


# ---------------------------------------------------------------------------
# 5. No brackets remain
# ---------------------------------------------------------------------------

class TestNoBracketsRemain:
    def test_all_non_passthrough_categories_replaced(self, pseudo):
        tokens = [
            "[PERSON-1]", "[DOCTOR-2]", "[HOSPITAL-1]", "[LOCATION-3]",
            "[DATE-1]", "[AGE-1]", "[PHONE-1]", "[EMAIL-1]", "[ID-1]",
            "[URL-1]", "[PATIENT]",
        ]
        text = " en ".join(tokens)
        rng = make_note_rng(0, "no_brackets_test")
        result = pseudo.pseudonymise(text, rng)
        assert re.search(r"\[[A-Z][A-Z_]*(?:-\d+)?\]", result.new_text) is None

    def test_passthrough_tokens_remain_verbatim(self, pseudo):
        text = "NYHA [NYHA] klasse en status [AFGEROND-1]."
        rng = make_note_rng(0, "passthrough_brackets_test")
        result = pseudo.pseudonymise(text, rng)
        assert "[NYHA]" in result.new_text
        assert "[AFGEROND-1]" in result.new_text


# ---------------------------------------------------------------------------
# 6. Passthrough categories
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_passthrough_token_left_verbatim(self, pseudo):
        rng = make_note_rng(0, "pt_verbatim")
        result = pseudo.pseudonymise("[NYHA]", rng)
        assert result.new_text == "[NYHA]"

    def test_passthrough_with_index_left_verbatim(self, pseudo):
        rng = make_note_rng(0, "pt_index")
        result = pseudo.pseudonymise("[AFGEROND-1]", rng)
        assert result.new_text == "[AFGEROND-1]"

    def test_passthrough_not_in_replacements(self, pseudo):
        rng = make_note_rng(0, "pt_replacements")
        result = pseudo.pseudonymise("status [AFGEROND-1] einde", rng)
        assert len(result.replacements) == 0

    def test_passthrough_emits_no_warning(self, pseudo, caplog):
        rng = make_note_rng(0, "pt_no_warn")
        logger_name = "nyha_temporal_identification.utils.pseudonymisation"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            pseudo.pseudonymise("[NYHA] en [TIA]", rng)
        assert len(caplog.records) == 0

    def test_passthrough_does_not_shift_offsets(self, pseudo):
        # [NYHA] is passthrough; [PERSON-1] after it is a real replacement.
        # The offset of text after [PERSON-1] must only account for [PERSON-1] length delta.
        text = "[NYHA] en [PERSON-1] klasse"
        rng = make_note_rng(0, "pt_offset")
        result = pseudo.pseudonymise(text, rng)
        # Exactly one replacement (PERSON-1), not two
        assert len(result.replacements) == 1
        # "klasse" starts at index 21 in original text ("[NYHA] en [PERSON-1] ")
        klasse_orig_start = text.index("klasse")
        klasse_orig_end = klasse_orig_start + len("klasse")
        ns, ne = result.remap(klasse_orig_start, klasse_orig_end)
        assert result.new_text[ns:ne] == "klasse"

    def test_all_passthrough_categories_defined(self):
        # Smoke-test that the frozenset is non-empty and contains expected entries
        assert "NYHA" in PASSTHROUGH_CATEGORIES
        assert "AFGEROND" in PASSTHROUGH_CATEGORIES
        assert "TIA" in PASSTHROUGH_CATEGORIES

    def test_custom_passthrough_overrides_default(self, data_dir):
        # Passing an empty passthrough means no token is skipped — NYHA gets fallback
        p = Pseudonymiser(data_dir=data_dir, passthrough=frozenset())
        rng = make_note_rng(0, "pt_custom")
        result = p.pseudonymise("[NYHA]", rng)
        assert result.new_text != "[NYHA]"


# ---------------------------------------------------------------------------
# 7. remap
# ---------------------------------------------------------------------------

class TestRemap:
    def _make_result(self, original: str, pseudo_: Pseudonymiser) -> PseudonymisationResult:
        rng = make_note_rng(0, "remap_test")
        return pseudo_.pseudonymise(original, rng)

    def test_no_replacement_before_span(self, pseudo):
        # No tokens before position 5; remap should return unchanged offsets.
        text = "Hello [PERSON-1] world"
        result = self._make_result(text, pseudo)
        # The person token starts at index 6 (after "Hello "), so remap(0, 5)
        # should not be affected.
        ns, ne = result.remap(0, 5)
        assert ns == 0
        assert ne == 5

    def test_single_replacement_shifts_offsets(self, pseudo):
        text = "[PERSON-1] NYHA II"
        result = self._make_result(text, pseudo)
        person_token = result.replacements[0]
        replacement_len = len(person_token.replacement)
        original_token_len = len(person_token.original_token)
        delta = replacement_len - original_token_len
        # The span "NYHA II" starts at 11 in original text (after "[PERSON-1] ")
        orig_start = 11
        orig_end = 18
        ns, ne = result.remap(orig_start, orig_end)
        assert ns == orig_start + delta
        assert ne == orig_end + delta
        # Verify by slicing
        assert result.new_text[ns:ne] == "NYHA II"

    def test_remap_span_containing_replacement(self, pseudo):
        text = "NYHA [PERSON-1] klasse"
        result = self._make_result(text, pseudo)
        # Span covers the whole text; remap(0, len(text)) should give (0, len(new_text))
        ns, ne = result.remap(0, len(text))
        assert ns == 0
        assert ne == len(result.new_text)

    def test_multi_replacement_cumulative_delta(self, pseudo):
        text = "[PERSON-1] bij [HOSPITAL-1] NYHA"
        result = self._make_result(text, pseudo)
        # Find where "NYHA" starts in original text
        nyha_orig_start = text.index("NYHA")
        nyha_orig_end = nyha_orig_start + 4
        ns, ne = result.remap(nyha_orig_start, nyha_orig_end)
        assert result.new_text[ns:ne] == "NYHA"


# ---------------------------------------------------------------------------
# 8. Synthetic generators
# ---------------------------------------------------------------------------

class TestSyntheticGenerators:
    def _rng(self) -> random.Random:
        return random.Random(0)

    def test_random_date_matches_date_pattern(self):
        rng = self._rng()
        combined = "|".join(f"(?:{p})" for p in DATE_PATTERNS)
        combined_re = re.compile(combined, re.IGNORECASE)
        for _ in range(20):
            d = random_date(rng)
            assert combined_re.search(d), f"Date {d!r} not matched by DATE_PATTERNS"

    def test_random_phone_shape(self):
        rng = self._rng()
        for _ in range(20):
            p = random_phone(rng)
            assert p.startswith("0"), f"Phone {p!r} does not start with 0"
            assert "-" in p

    def test_random_email_shape(self):
        rng = self._rng()
        for _ in range(20):
            e = random_email(rng)
            assert "@example.nl" in e

    def test_random_id_digits_only(self):
        rng = self._rng()
        for _ in range(20):
            i = random_id(rng)
            assert i.isdigit(), f"ID {i!r} contains non-digits"
            assert 7 <= len(i) <= 10

    def test_data_backed_generators_return_strings(self, data_dir):
        p = Pseudonymiser(data_dir=data_dir)
        rng = random.Random(1)
        text = "[PERSON-1] [HOSPITAL-2] [LOCATION-3]"
        result = p.pseudonymise(text, rng)
        for r in result.replacements:
            assert isinstance(r.replacement, str)
            assert len(r.replacement) > 0


# ---------------------------------------------------------------------------
# 9. Integration: find_nyha_class with pseudonymiser
# ---------------------------------------------------------------------------

class TestFindNyhaClassIntegration:
    def test_pseudonymisation_rewrites_tokens(self, nlp, pseudo):
        from nyha_temporal_identification.extraction.parsing import find_nyha_class

        text = "De patiënt heeft NYHA [PERSON-1] klachten."
        rng = make_note_rng(42, "integration_test")
        result = find_nyha_class(text, nlp_pipeline=nlp, pseudonymiser=pseudo, rng=rng)

        assert result.pseudonymised_text is not None
        assert "[PERSON-1]" not in result.pseudonymised_text

    def test_invalid_nyha_mention_flagged_and_offsets_correct(self, nlp, pseudo):
        from nyha_temporal_identification.extraction.parsing import find_nyha_class

        text = "De patiënt heeft NYHA [PERSON-1] klachten."
        rng = make_note_rng(42, "integration_offset_test")
        result = find_nyha_class(text, nlp_pipeline=nlp, pseudonymiser=pseudo, rng=rng)

        assert len(result.extracted_invalid_nyha_mentions) == 1
        mention = result.extracted_invalid_nyha_mentions[0]
        pseudo_text = result.pseudonymised_text
        # Offsets must slice correctly from the pseudonymised text
        assert pseudo_text is not None
        sliced = pseudo_text[mention.start:mention.end]
        assert sliced == mention.original_text

    def test_valid_nyha_mention_survives(self, nlp, pseudo):
        from nyha_temporal_identification.extraction.parsing import find_nyha_class

        text = "Patiënt heeft NYHA II klachten en NYHA [DOCTOR-1] eerder."
        rng = make_note_rng(42, "integration_valid_test")
        result = find_nyha_class(text, nlp_pipeline=nlp, pseudonymiser=pseudo, rng=rng)

        # The valid NYHA II mention must be extracted
        assert len(result.extracted_nyha_mentions) >= 1
        classes = [m.classes for m in result.extracted_nyha_mentions]
        assert any(2 in c for c in classes), f"No class-II mention found; got {classes}"

    def test_disabled_pseudonymisation_unchanged(self, nlp):
        from nyha_temporal_identification.extraction.parsing import find_nyha_class

        text = "De patiënt heeft NYHA II klachten."
        result = find_nyha_class(text, nlp_pipeline=nlp)
        assert result.pseudonymised_text is None
        assert len(result.extracted_nyha_mentions) == 1
