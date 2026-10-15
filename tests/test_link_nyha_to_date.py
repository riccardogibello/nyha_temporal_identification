"""Unit tests for link_nyha_to_date in extraction/parsing.py.

link_nyha_to_date takes a list[ParsedDate] and a spaCy Span (accessed only via
.start_char / .end_char), so we stub the span with SimpleNamespace to keep
these tests self-contained.
"""
import types

import pytest

from nyha_temporal_identification.constants import DEFAULT_NO_INTEGER_VALUE
from nyha_temporal_identification.data_model.general import ParsedDate
from nyha_temporal_identification.extraction.parsing import link_nyha_to_date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date(start: int, end: int, label: str = "2023-01-01") -> ParsedDate:
    return ParsedDate(text="test", parsed_date=label, start=start, end=end)


def _span(start_char: int, end_char: int):
    """Minimal stand-in for a spaCy Span."""
    return types.SimpleNamespace(start_char=start_char, end_char=end_char)


# ---------------------------------------------------------------------------
# TestLinkNyhaToDate
# ---------------------------------------------------------------------------


class TestLinkNyhaToDate:
    # -- single backward candidate ------------------------------------------

    def test_backward_within_distance(self):
        dates = [_date(0, 10)]
        result, dist = link_nyha_to_date(dates, _span(20, 30), max_distance=15)
        assert result is not None
        assert dist == 10  # span.start_char − date.end = 20 − 10

    def test_backward_beyond_distance_not_linked(self):
        dates = [_date(0, 10)]
        result, dist = link_nyha_to_date(dates, _span(100, 110), max_distance=15)
        assert result is None
        assert dist == DEFAULT_NO_INTEGER_VALUE

    def test_distance_exactly_at_max_is_linked(self):
        # dist = 25 − 10 = 15 == max_distance → should link (≤)
        dates = [_date(0, 10)]
        result, dist = link_nyha_to_date(dates, _span(25, 35), max_distance=15)
        assert result is not None
        assert dist == 15

    def test_distance_one_beyond_max_not_linked(self):
        # dist = 26 − 10 = 16 > 15 → should not link
        dates = [_date(0, 10)]
        result, dist = link_nyha_to_date(dates, _span(26, 36), max_distance=15)
        assert result is None
        assert dist == DEFAULT_NO_INTEGER_VALUE

    # -- forward candidates -------------------------------------------------

    def test_forward_date_ignored_when_only_backward(self):
        dates = [_date(50, 60)]
        result, _ = link_nyha_to_date(
            dates, _span(10, 20), max_distance=200, only_backward=True
        )
        assert result is None

    def test_forward_date_linked_when_only_backward_false(self):
        dates = [_date(50, 60)]
        result, dist = link_nyha_to_date(
            dates, _span(10, 20), max_distance=200, only_backward=False
        )
        assert result is not None
        assert dist == 30  # date.start − span.end_char = 50 − 20

    def test_forward_date_beyond_max_not_linked(self):
        dates = [_date(100, 110)]
        result, _ = link_nyha_to_date(
            dates, _span(0, 10), max_distance=50, only_backward=False
        )
        assert result is None  # dist = 100 − 10 = 90 > 50

    # -- empty / no candidates ----------------------------------------------

    def test_empty_dates_returns_none_and_sentinel(self):
        result, dist = link_nyha_to_date([], _span(10, 20), max_distance=100)
        assert result is None
        assert dist == DEFAULT_NO_INTEGER_VALUE

    # -- multiple candidates: closest wins ----------------------------------

    def test_multiple_backward_dates_closest_wins(self):
        far = _date(0, 5, "2020-01-01")    # dist = 50 − 5 = 45
        close = _date(40, 45, "2022-01-01")  # dist = 50 − 45 = 5
        result, dist = link_nyha_to_date([far, close], _span(50, 60), max_distance=200)
        assert result is close
        assert dist == 5

    def test_tie_distance_first_encountered_wins(self):
        """Strict < comparison means equal-distance candidates keep the first."""
        d1 = _date(0, 5, "2020-01-01")
        d2 = _date(0, 5, "2021-01-01")
        result, dist = link_nyha_to_date([d1, d2], _span(50, 60), max_distance=200)
        assert result is d1
        assert dist == 45

    def test_forward_closer_than_backward_wins(self):
        backward = _date(0, 5, "2020-01-01")     # dist = 50 − 5 = 45
        forward = _date(65, 70, "2022-01-01")    # dist = 65 − 60 = 5
        result, dist = link_nyha_to_date(
            [backward, forward], _span(50, 60), max_distance=200, only_backward=False
        )
        assert result is forward
        assert dist == 5

    def test_both_directions_within_distance_picks_closest(self):
        backward = _date(30, 40, "2021-01-01")   # dist = 50 − 40 = 10
        forward = _date(75, 80, "2022-01-01")    # dist = 75 − 60 = 15
        result, dist = link_nyha_to_date(
            [backward, forward], _span(50, 60), max_distance=200, only_backward=False
        )
        assert result is backward
        assert dist == 10

    # -- side effects: consumed dates removed from list ---------------------

    def test_linked_date_removed_from_list(self):
        dates = [_date(0, 10)]
        link_nyha_to_date(dates, _span(20, 30), max_distance=100)
        assert dates == []

    def test_only_linked_date_removed_not_all(self):
        far = _date(0, 5, "2020-01-01")
        close = _date(40, 45, "2022-01-01")
        dates = [far, close]
        link_nyha_to_date(dates, _span(50, 60), max_distance=200)
        # close was matched and consumed; far stays
        assert len(dates) == 1
        assert dates[0] is far

    def test_unlinked_date_stays_in_list(self):
        dates = [_date(0, 10)]
        link_nyha_to_date(dates, _span(200, 210), max_distance=15)
        assert len(dates) == 1

    def test_no_match_leaves_list_unchanged(self):
        dates = [_date(500, 510), _date(600, 610)]
        link_nyha_to_date(dates, _span(0, 10), max_distance=50)
        assert len(dates) == 2

    # -- return value structure ---------------------------------------------

    def test_returns_tuple_of_two(self):
        ret = link_nyha_to_date([], _span(10, 20))
        assert isinstance(ret, tuple)
        assert len(ret) == 2

    def test_parsed_date_object_returned(self):
        expected = _date(0, 10, "2023-06-01")
        result, _ = link_nyha_to_date([expected], _span(20, 30), max_distance=100)
        assert result is expected
