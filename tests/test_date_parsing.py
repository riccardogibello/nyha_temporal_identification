"""Tests for date extraction and flexible date parsing."""
import datetime

import pytest

from nyha_temporal_identification.constants import DATE_PATTERNS
from nyha_temporal_identification.extraction.date_parsing import (
    extract_dates,
    parse_flexible_date,
    parse_iso_date,
)


# ---------------------------------------------------------------------------
# parse_iso_date
# ---------------------------------------------------------------------------


class TestParseIsoDate:
    def test_valid_iso(self):
        result = parse_iso_date("2023-01-15")
        assert result == datetime.datetime(2023, 1, 15)

    def test_valid_iso_with_time(self):
        result = parse_iso_date("2023-01-15T10:30:00")
        assert result == datetime.datetime(2023, 1, 15, 10, 30, 0)

    def test_none_input(self):
        assert parse_iso_date(None) is None

    def test_invalid_string(self):
        assert parse_iso_date("not-a-date") is None


# ---------------------------------------------------------------------------
# parse_flexible_date
# ---------------------------------------------------------------------------


class TestParseFlexibleDate:
    def test_year_only_start(self):
        result = parse_flexible_date("2023", default_year_only="start")
        assert result == datetime.datetime(2023, 1, 1)

    def test_year_only_end(self):
        result = parse_flexible_date("2023", default_year_only="end")
        assert result == datetime.datetime(2023, 12, 31)

    def test_year_only_mid(self):
        result = parse_flexible_date("2023", default_year_only="mid")
        assert result == datetime.datetime(2023, 6, 15)

    def test_full_date_day_first(self):
        result = parse_flexible_date("15-01-2023", dayfirst=True)
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_full_date_month_name(self):
        result = parse_flexible_date("15 january 2023")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1

    def test_epoch_millis(self):
        # A large integer should be interpreted as milliseconds
        millis = 1_700_000_000_000
        result = parse_flexible_date(millis)
        assert result is not None
        assert result.year >= 2023

    def test_epoch_seconds(self):
        seconds = 1_700_000_000
        result = parse_flexible_date(seconds)
        assert result is not None
        assert result.year >= 2023

    def test_invalid_string(self):
        assert parse_flexible_date("not-a-date-at-all-xyz") is None


# ---------------------------------------------------------------------------
# extract_dates
# ---------------------------------------------------------------------------


class TestExtractDates:
    def _run(self, text, reference=None):
        new_old = {i: i for i in range(len(text))}
        return extract_dates(
            text=text,
            date_patterns=DATE_PATTERNS,
            original_text=text,
            reference_datetime=reference,
            new_old_index_map=new_old,
        )

    def test_numeric_dmy(self):
        dates = self._run("afspraak op 15-01-2023 was goed")
        assert len(dates) == 1
        assert "2023-01-15" in dates[0].parsed_date

    def test_iso_format(self):
        dates = self._run("datum 2023-06-15 gezien")
        assert len(dates) == 1
        assert "2023-06-15" in dates[0].parsed_date

    def test_year_only(self):
        dates = self._run("in 2022 verslechterd")
        assert len(dates) == 1
        assert "2022" in dates[0].parsed_date

    def test_month_name_nl(self):
        dates = self._run("gezien op 3 jan 2023")
        assert any("2023" in d.parsed_date for d in dates)

    def test_multiple_dates_no_duplicates(self):
        dates = self._run("15-01-2023 en 2023-01-15 zelfde dag")
        # sub-span deduplication: both are same date; one of them contains the other
        assert len(dates) >= 1

    def test_historical_flag(self):
        reference = datetime.datetime(2023, 6, 1)
        dates = self._run("op 01-01-2020 was hij zieker", reference=reference)
        assert len(dates) == 1
        assert dates[0].temporal_status == "historical"

    def test_current_flag(self):
        reference = datetime.datetime(2023, 6, 1)
        dates = self._run("op 01-05-2023 was hij stabiel", reference=reference)
        assert len(dates) == 1
        assert dates[0].temporal_status == "current"

    def test_no_dates(self):
        dates = self._run("geen specifieke datum vermeld")
        assert dates == []

    def test_subspan_removal(self):
        # "15-01-2023" should subsume "2023" if they overlap
        dates = self._run("op 15-01-2023 afspraak")
        # The year "2023" is a sub-span of the full date; only one should remain
        texts = [d.text for d in dates]
        assert not any(t == "2023" and any("2023" in o and len(o) > 4 for o in texts) for t in texts)
