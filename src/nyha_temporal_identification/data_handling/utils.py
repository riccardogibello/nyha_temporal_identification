import re
from datetime import datetime

from nyha_temporal_identification.constants import NO_LINKED_DATE_TAG, SPECIAL_IDS


def is_special_entity(
    identifier: str,
    text: str,
):
    id_in_specials = identifier in SPECIAL_IDS
    text_in_specials = text in [
        "NO_LINKED_DATE",
        "CURRENT_DATE",
        "HISTORICAL_DATE",
    ]

    return id_in_specials or text_in_specials


def translate_nyha_name(
    nyha_name: str,
) -> tuple[list[int], list[int]]:
    if nyha_name == "nyha class 1":
        return [1], [1, 0, 0, 0]
    elif nyha_name == "nyha class 1-2":
        return [1, 2], [1, 1, 0, 0]
    elif nyha_name == "nyha class 2":
        return [2], [0, 1, 0, 0]
    elif nyha_name == "nyha class 2-3":
        return [2, 3], [0, 1, 1, 0]
    elif nyha_name == "nyha class 3":
        return [3], [0, 0, 1, 0]
    elif nyha_name == "nyha class 3-4":
        return [3, 4], [0, 0, 1, 1]
    elif nyha_name == "nyha class 4":
        return [4], [0, 0, 0, 1]
    else:
        raise ValueError(f"Unknown NYHA class name: {nyha_name}")


def extract_authored_date(text: str) -> datetime | None:
    """
    Parses a long string to find the AUTHORED date pattern and returns it as a datetime object.

    Example pattern:
        76\n\nAUTHORED = 2016-01-10T23:00:00\n\nTYPE     = Brief\n==

    Returns:
        datetime object if found, None otherwise.
    """
    pattern = r"AUTHORED\s*=\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    match = re.search(pattern, text)

    if match:
        date_str = match.group(1)
        return datetime.fromisoformat(date_str)

    return None


def remove_extra_text(
    text: str,
) -> str:
    """
    This method removes the extra text added at the end of the deidentified text of a clinical note,
    which was used to provide additional information to annotators.

    Args:
        text (str): The input text containing the deidentified clinical note with potential extra information.

    Returns:
        str: The cleaned text with the extra information removed.
    """
    # Find the NO_LINKED_DATE subtext, split, and return the first part
    if NO_LINKED_DATE_TAG in text:
        return text.split(NO_LINKED_DATE_TAG)[0]
    return text
