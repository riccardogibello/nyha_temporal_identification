import re
from datetime import datetime

from dateutil import parser

from nyha_temporal_identification.constants import (
    DEFAULT_NO_INTEGER_VALUE,
    HISTORICAL_DAYS_THRESHOLD,
    NYHA_CURRENT_STATUS,
    NYHA_HISTORICAL_STATUS,
)
from nyha_temporal_identification.data_model.general import ParsedDate


def _remove_subspans(
    spans: list[ParsedDate],
) -> list[ParsedDate]:
    # Sort by start position of the span, then by longest span first
    spans = sorted(spans, key=lambda x: (x.start, -(x.end - x.start)))

    filtered = []
    for span in spans:
        # If the current span is not fully contained within any of the kept spans, keep it
        if not any(
            span.start >= kept.start and span.end <= kept.end for kept in filtered
        ):
            filtered.append(span)

    return filtered


def parse_iso_date(
    date_str: str | None = None,
) -> datetime | None:
    """
    Parse an ISO 8601 date string into a datetime object.
    Args:
        date_str: The ISO 8601 date string to parse.
    Returns:
        A datetime object if parsing was successful, None otherwise.
    """
    if date_str is None:
        return None

    try:
        dt = datetime.fromisoformat(date_str)
        return dt
    except ValueError:
        return None


def parse_flexible_date(
    date_str: str,
    *,
    dayfirst: bool = True,
    default_year_only: str = "start",
) -> datetime | None:
    """
    Convert in a flexible manner a date string into a datetime object.

    Returns:
        A datetime object if parsing was successful, None otherwise.
    """

    if type(date_str) is int or type(date_str) is float:
        val = float(date_str)
        # If the value is very large (e.g. > year 5138 in seconds), interpret it as milliseconds
        if val > 1e11:
            val /= 1000.0

        # Convert the epoch to datetime
        try:
            return datetime.fromtimestamp(val)
        except (OSError, OverflowError, ValueError):
            return None
    date_str = date_str.strip()

    # Look for year-only format
    if date_str.isdigit() and len(date_str) == 4:
        year = int(date_str)

        if default_year_only == "start":
            return datetime(year, 1, 1)
        elif default_year_only == "end":
            return datetime(year, 12, 31)
        else:  # "mid"
            return datetime(year, 6, 15)

    # Otherwise, parse the date using dateutil.parser
    try:
        dt = parser.parse(
            date_str,
            dayfirst=dayfirst,
            yearfirst=False,
            default=datetime(1900, 1, 1),
        )
        return dt
    except (ValueError, OverflowError):
        return None


def extract_dates(
    text: str,
    date_patterns: list[str],
    original_text: str,
    reference_datetime: datetime | None = None,
    new_old_index_map: dict[int, int] | None = None,
) -> list[ParsedDate]:
    dates = []

    # For each given date pattern, find all matches in the text
    for pat in date_patterns:
        for m in re.finditer(
            pat,
            text,
        ):
            date_text = m.group()
            current_date = parse_flexible_date(
                date_text,
            )
            # If the timespan between the reference and current dates
            # if greater than the selected threshold, mark as historical
            is_hist = False
            if reference_datetime is not None and current_date is not None:
                delta = reference_datetime - current_date
                if abs(delta.days) > HISTORICAL_DAYS_THRESHOLD:
                    is_hist = True

            original_start = None
            original_end = None
            if new_old_index_map is not None:
                # Get the old string start and end positions
                original_start = new_old_index_map[m.start()]
                original_end = new_old_index_map[m.end() - 1] + 1
            # Get the original text span
            original_date_text = original_text[original_start:original_end]

            # Create a new date instance
            date_instance = ParsedDate(
                text=original_date_text,
                parsed_date=current_date.isoformat()
                if current_date is not None
                else None,
                original_start=original_start
                if original_start is not None
                else DEFAULT_NO_INTEGER_VALUE,
                original_end=original_end
                if original_end is not None
                else DEFAULT_NO_INTEGER_VALUE,
                temporal_status=NYHA_HISTORICAL_STATUS
                if is_hist
                else NYHA_CURRENT_STATUS,
                start=m.start(),
                end=m.end(),
            )
            dates.append(date_instance)

    dates = _remove_subspans(
        dates,
    )
    return dates
