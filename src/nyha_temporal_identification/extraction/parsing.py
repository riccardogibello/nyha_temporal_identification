import datetime
import json
import logging
import os
import re
from typing import Tuple, Union

import numpy as np
import spacy
from datasets import Dataset
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from nyha_temporal_identification.constants import (
    AUTHORED_COLUMN,
    CURRENT_NYHA_DATE_CHAR_DISTANCE,
    DATE_PATTERNS,
    DEFAULT_CPUS_PER_TASK,
    DEFAULT_NO_INTEGER_VALUE,
    DEIDENTIFIED_TEXT_COLUMN,
    EXTRACTED_DATES_KEY,
    EXTRACTED_INVALID_NYHA_MENTIONS_KEY,
    EXTRACTED_NYHA_MENTIONS_KEY,
    ID_COLUMN,
)
from nyha_temporal_identification.data_model.general import (
    InvalidNyhaMention,
    Mentions,
    NyhaMention,
    ParsedDate,
)
from nyha_temporal_identification.extraction.date_parsing import (
    extract_dates,
    parse_flexible_date,
)
from nyha_temporal_identification.extraction.note import (
    build_final_note_map,
    preprocess_notes,
    structure_note,
)
from nyha_temporal_identification.utils.text_cleaning import (
    LINE_BREAK_SENTINEL,
    clean_text,
)


def norm_token(
    tok: str,
    remove_all=False,
) -> str:
    for sym in ["-", " "]:
        tok = re.sub(rf"{sym}+", "" if remove_all else sym, tok)
    return tok.lower().replace("l", "i")


def parse_single(
    tok: str,
    remove_all=False,
) -> Union[np.ndarray, None]:
    s = norm_token(tok, remove_all)
    single_map = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "een": 1,
        "eén": 1,
        "twee": 2,
        "drie": 3,
        "vier": 4,
    }
    idx = single_map.get(s, 0)
    if idx:
        arr = np.zeros(4, dtype=int)
        arr[idx - 1] = 1
        return arr
    return None


def parse_pair_or_single(
    raw: str,
):
    single = parse_single(raw)
    if single is not None:
        return single

    # Slash separator is allowed for all pairs except III/IV
    has_slash = bool(re.search(r"/", raw))

    # Replace any separator (spaces, hyphens, slashes, "of", "or") with a single hyphen
    collapsed = re.sub(
        r"[\s\-\/]+|(?:\s*of\s*|\s*or\s*)", "-", raw, flags=re.IGNORECASE
    ).strip("-")
    if "-" in collapsed:
        a, b = collapsed.split("-", 1)
        va, vb = parse_single(a, remove_all=True), parse_single(b, remove_all=True)
        if va is not None and vb is not None:
            va = np.where(va == 1)[0][0] + 1
            vb = np.where(vb == 1)[0][0] + 1
            # Unified "collapse to single class" rule (see
            # docs/spec/nyha_class_vector_convention.md).
            #
            # Reversed ranges (e.g. IV-I, 4-1) are not clinically meaningful.
            if va > vb:
                return None
            # Same class repeated (e.g. "II-II") collapses to a single class.
            if va == vb:
                result = np.zeros(4, dtype=int)
                result[va - 1] = 1
                return result
            # Adjacent pair (I-II, II-III, III-IV): a genuine range, except
            # "III/IV" which is ambiguous with the "III on a scale of IV" reading.
            if vb - va == 1:
                if va == 3 and has_slash:
                    return None
                result = np.zeros(4, dtype=int)
                result[va - 1] = 1
                result[vb - 1] = 1
                return result
            # Broad range whose upper bound is the scale maximum IV (I-IV, II-IV):
            # read "-IV" as the scale denominator and collapse to the lower class.
            if vb == 4 and va in (1, 2):
                result = np.zeros(4, dtype=int)
                result[va - 1] = 1
                return result
            # I-III and any other non-adjacent, non-scale pair is invalid.
            return None
    return None


Span.set_extension("nyha_vector", default=None, force=True)
Span.set_extension("nyha_classes", default=None, force=True)

# Matches NYHA (or full-name variants) followed by an optional class-word token and
# then directly by a de-anonymization placeholder, with NO class numeral in between.
# This is the fingerprint of the de-identification pipeline replacing a Roman numeral
# (I, II, III, IV) with a placeholder like [PERSON-1] or [DOCTOR-2].
# Cases where a numeral IS present (e.g. "NYHA II [DOCTOR-1]") do NOT match because
# "II" is not consumed by the optional class-token group, blocking the \[ anchor.
_INVALID_NYHA_DEANON_RE = re.compile(
    r"(?<![a-zA-Z0-9])"
    r"(?:nyha|new\s+york\s+(?:heart|hart)(?:\s+(?:association|associatie|vereniging|svereniging))?)"
    r"[\s\-/]*"
    r"(?:(?:class|classificatie|klasse|kl\.?|c\.?)[\s\-/]*)?"
    r"\[[A-Z][A-Z_]*-\d+\]",
    re.IGNORECASE,
)


def _build_nyha_patterns() -> list[str]:
    """Return the three NYHA regex patterns used by nyha_component and find_raw_nyha_spans."""
    sep = r"(?:[\s\-/\x00]| of | or )*"

    value_single = (
        r"(?:[il]{1,3}|iv|lv(?!\s+[a-z])|1|2|3|4|one|two|three|four|een|eén|twee|drie|vier)"
    )

    ny_en = rf"new{sep}york{sep}heart{sep}association"
    ny_nl = rf"new{sep}york{sep}hart(?:{sep}(?:associatie|svereniging|vereniging))?"
    class_tokens = r"(?:class|classificatie|klasse|kl|c|kl\.|c\.)*"

    roman_single = r"(?:[il]{1,3}|iv|lv(?!\s+[a-z]))"

    value_pair_group = (
        rf"({value_single}"
        rf"(?:(?:\s*[-/]\s*|\s+(?:of|or)\s+){value_single}"
        rf"|\s+{roman_single})?)"
    )

    return [
        rf"(?<![a-z0-9])(?:{ny_en}|{ny_nl}){sep}{class_tokens}{sep}{value_pair_group}(?![a-z0-9])",
        rf"(?<![a-z0-9])nyha{sep}{class_tokens}{sep}{value_pair_group}(?![a-z0-9])",
        rf"(?<![a-z0-9]){class_tokens}{sep}{value_pair_group}{sep}nyha(?![a-z0-9])",
    ]


def find_raw_nyha_spans(text: str) -> list[tuple[int, int]]:
    """
    Find all NYHA-pattern intervals in raw text without spaCy tokenisation or cleaning.

    Runs the three NYHA patterns from _build_nyha_patterns() directly on the
    text (case-insensitive).  Overlapping intervals are merged into maximal
    intervals.  parse_pair_or_single is NOT called — invalid values are returned
    too.  Newlines are matched by \\s in the separator, so pairs can cross line
    boundaries (intentional over-match compared to nyha_component).

    Returns a list of (start, end) tuples in the original text space.
    """
    intervals: list[tuple[int, int]] = []
    for pat in _build_nyha_patterns():
        for m in re.finditer(pat, text, re.IGNORECASE):
            intervals.append((m.start(), m.end()))

    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


@Language.component("nyha_component")
def nyha_component(
    doc: Doc,
) -> Doc:
    # Use the sentinel text (injected by find_nyha_class) when available.
    # The sentinel text has the same character positions as doc.text but
    # replaces newlines with \x00, preventing value_sep from absorbing
    # next-line tokens into a pair while still allowing bare spaces.
    text = doc.user_data.get("sentinel_text", doc.text)
    spans = []

    patterns = _build_nyha_patterns()

    for pat in patterns:
        for m in re.finditer(pat, text):
            value = parse_pair_or_single(m.group(1))
            if value is None:
                logging.debug(f"Skipping invalid NYHA value for match: {m.group(0)}")
                continue

            # "expand" snaps the end boundary to the nearest token end
            # when the regex match terminates mid-token (e.g. the value
            # group ends at "-" or "+" that is fused into the same spaCy
            # token as the class numeral).  The extracted class comes from
            # m.group(1), not from the span text, so expansion is harmless.
            span = doc.char_span(
                m.start(),
                m.end(),
                label="NYHA",
                alignment_mode="expand",
            )
            if span is None:
                logging.debug(f"Skipping invalid span for match: {m.group(0)}")
                continue

            span._.nyha_vector = value
            span._.nyha_classes = [int(i + 1) for i, v in enumerate(value) if v == 1]
            spans.append(span)

    doc.ents = filter_spans(
        list(doc.ents) + spans,
    )
    return doc


def link_nyha_to_date(
    dates: list[ParsedDate],
    nyha_span,
    max_distance=250,
    only_backward=True,
) -> Tuple[Union[ParsedDate, None], Union[int, None]]:
    """
    This method links a given NYHA mention to the nearest date found in the text
    within a specified character distance, considering only the previous dates by
    default, and returns the linked date information along with the character
    distance between the NYHA mention and the linked date. If no date is found
    within the specified distance, it returns None for the date information and
    the default maximum distance.

    Args:
        dates (list[ParsedDate]): A list of ParsedDate objects containing the extracted date
        information from the text, with their original character positions.
        nyha_span (_type_): The Span object representing the NYHA mention
        for which to find the nearest date.
        max_distance (int, optional): The maximum character distance to consider
        for linking a date to the NYHA mention. Defaults to 250.
        only_backward (bool, optional): Whether to consider only the dates
        that appear before the NYHA mention for linking. Defaults to True.

    Returns:
        Tuple[Union[ParsedDate, None], Union[int, None]]: A tuple containing the linked
        date information and the character distance.
    """
    best: ParsedDate | None = None
    best_dist = int(CURRENT_NYHA_DATE_CHAR_DISTANCE)
    linked_index = DEFAULT_NO_INTEGER_VALUE
    for index, d in enumerate(dates):
        # If it's before the NYHA span, calculate the distance
        if d.end <= nyha_span.start_char:
            dist = nyha_span.start_char - d.end
        # If it's after the NYHA span, calculate the distance only if it is allowed
        # to look forward
        elif not only_backward and d.start >= nyha_span.end_char:
            dist = d.start - nyha_span.end_char
        else:
            continue

        # If the found distance is within the maximum allowed distance,
        if dist <= max_distance:
            # and it is better than the best found so far, keep it
            if best_dist == DEFAULT_NO_INTEGER_VALUE or dist < best_dist:
                best_dist = dist
                best = d
                linked_index = index

    # Remove from the list the linked date to avoid multiple NYHA mentions linking to the same date
    if linked_index >= 0:
        dates.pop(linked_index)

    # Return DEFAULT_NO_INTEGER_VALUE when no date was linked so callers can
    # distinguish "no match" from a genuine distance of 0.
    return best, best_dist if best is not None else DEFAULT_NO_INTEGER_VALUE


def find_nyha_class(
    text: str,
    authored_datetime: Union[datetime.datetime, None] = None,
    nlp_pipeline=None,
    pseudonymiser=None,
    rng=None,
) -> Mentions:
    """
    This method returns a map containing the NYHA metions found in the given text, with their possible
    linked date information, and the extracted dates that were not linked to any NYHA mention.

    Args:
        text (str): The original text to analyze for NYHA mentions and dates.
        authored_datetime (Union[datetime.datetime, None], optional): The reference datetime to use
        for relative date parsing. Defaults to None.
        nlp_pipeline (_type_, optional): The spacy NLP pipeline to use for processing the text. Defaults to None.
        pseudonymiser: Optional Pseudonymiser instance. When provided together with rng, bracket
        tokens in text are replaced before extraction.
        rng: Per-note random.Random instance for deterministic pseudonymisation.

    Raises:
        ValueError: A ValueError is raised if the nlp_pipeline argument is None, indicating that a
        spacy pipeline instance must be provided.

    Returns:
        Mentions: An object containing the extracted NYHA mentions and dates.
    """
    if nlp_pipeline is None:
        raise ValueError(
            "nlp_pipeline cannot be None. Please provide a spacy pipeline instance."
        )

    pseudonymised_text: str | None = None

    if pseudonymiser is not None and rng is not None:
        # Step 0a: scan raw token-bearing text for invalid-NYHA patterns BEFORE rewriting,
        # so the de-identification placeholder is still visible as the signal.
        raw_invalid_matches = list(_INVALID_NYHA_DEANON_RE.finditer(text))
        # Step 0b: replace all bracket tokens with realistic values.
        pseudo_result = pseudonymiser.pseudonymise(text, rng)
        working_text = pseudo_result.new_text
        pseudonymised_text = working_text
        # Step 0c: remap invalid-mention offsets into pseudonymised text space.
        invalid_nyha_info: list[InvalidNyhaMention] = []
        for m in raw_invalid_matches:
            ns, ne = pseudo_result.remap(m.start(), m.end())
            span_text = working_text[ns:ne]
            invalid_nyha_info.append(
                InvalidNyhaMention(
                    original_text=span_text,
                    cleaned_text=span_text.lower(),
                    start=ns,
                    end=ne,
                    original_start=ns,
                    original_end=ne,
                )
            )
    else:
        working_text = text
        invalid_nyha_info = [
            InvalidNyhaMention(
                original_text=text[m.start() : m.end()],
                cleaned_text=text[m.start() : m.end()].lower(),
                start=m.start(),
                end=m.end(),
                original_start=m.start(),
                original_end=m.end(),
            )
            for m in _INVALID_NYHA_DEANON_RE.finditer(text)
        ]

    # Clean the given text and get a map to backtrack character indexes to the original text
    cleaned_text, new_old_index_map = clean_text(
        working_text,
        normalize_text=True,
    )
    # Give spaCy a space-only version so its tokeniser produces clean token
    # boundaries (spaCy does not split on \x00).  The sentinel text is injected
    # via user_data so nyha_component can still use it for the regex; character
    # positions are identical because \x00 and ' ' both occupy one byte.
    spacy_text = cleaned_text.replace(LINE_BREAK_SENTINEL, " ")
    doc = nlp_pipeline.make_doc(spacy_text)
    doc.user_data["sentinel_text"] = cleaned_text
    for _, pipe in nlp_pipeline.pipeline:
        doc = pipe(doc)

    # Extract dates from the text
    dates: list[ParsedDate] = extract_dates(
        text=cleaned_text,
        date_patterns=DATE_PATTERNS,
        reference_datetime=authored_datetime,
        new_old_index_map=new_old_index_map,
        original_text=working_text,
    )

    # Extract NYHA info, along with the linked date info if present in the specified character distance,
    # and the original text positions of each of them
    nyha_info: list[NyhaMention] = []
    for ent in doc.ents:
        if ent.label_ == "NYHA":
            # Get the original start and end positions
            original_start = new_old_index_map[ent.start_char]
            original_end = new_old_index_map[ent.end_char - 1] + 1

            # Search for the nearest date to the NYHA mention, considering
            # only the previous ones; note that each date is removed from the
            # list once it is linked to a NYHA mention to avoid multiple linkings
            linked_date, linked_date_distance = link_nyha_to_date(
                dates,
                ent,
                max_distance=int(CURRENT_NYHA_DATE_CHAR_DISTANCE),
                only_backward=True,
            )

            char_distance = (
                linked_date_distance
                if linked_date_distance is not None
                and linked_date_distance != DEFAULT_NO_INTEGER_VALUE
                else None
            )
            nyha_mention = NyhaMention(
                original_text=working_text[original_start:original_end],
                cleaned_text=ent.text,
                classes=ent._.nyha_classes,
                one_hot_vector=[int(v) for v in ent._.nyha_vector.tolist()],
                start=int(ent.start_char),
                end=int(ent.end_char),
                original_start=int(original_start),
                original_end=int(original_end),
                linked_date=linked_date,
                char_distance=char_distance,
            )
            nyha_info.append(nyha_mention)

    return Mentions(
        extracted_nyha_mentions=nyha_info,
        extracted_invalid_nyha_mentions=invalid_nyha_info,
        extracted_dates=dates,
        pseudonymised_text=pseudonymised_text,
    )


def find_nyha_and_dates(
    batch: dict,
    nlp_pipeline: Language,
    pseudonymiser=None,
    seed: int | None = None,
) -> dict:
    """Clean text and extract NYHA mentions and dates for a batch."""
    from nyha_temporal_identification.utils.pseudonymisation import make_note_rng

    extracted_mentions_batch: list[str] = []
    extracted_invalid_mentions_batch: list[str] = []
    extracted_dates_batch: list[str] = []
    pseudonymised_texts_batch: list[str] = []

    ids = batch[ID_COLUMN]
    texts = batch[DEIDENTIFIED_TEXT_COLUMN]
    authoreds = batch.get(AUTHORED_COLUMN, [None] * len(ids))

    for note_id, text, authored in zip(ids, texts, authoreds):
        rng = make_note_rng(seed, note_id) if pseudonymiser is not None and seed is not None else None
        nyha_res = find_nyha_class(
            text=text,
            authored_datetime=(
                parse_flexible_date(
                    authored,
                )
                if authored
                else None
            ),
            nlp_pipeline=nlp_pipeline,
            pseudonymiser=pseudonymiser,
            rng=rng,
        )
        extracted_mentions_batch.append(
            json.dumps(
                [m.get_json_serializable() for m in nyha_res.extracted_nyha_mentions]
            )
        )
        extracted_invalid_mentions_batch.append(
            json.dumps(
                [
                    m.get_json_serializable()
                    for m in nyha_res.extracted_invalid_nyha_mentions
                ]
            )
        )
        extracted_dates_batch.append(
            json.dumps([d.get_json_serializable() for d in nyha_res.extracted_dates])
        )
        if pseudonymiser is not None:
            pseudonymised_texts_batch.append(nyha_res.pseudonymised_text or text)

    result: dict = {
        EXTRACTED_NYHA_MENTIONS_KEY: extracted_mentions_batch,
        EXTRACTED_INVALID_NYHA_MENTIONS_KEY: extracted_invalid_mentions_batch,
        EXTRACTED_DATES_KEY: extracted_dates_batch,
    }
    if pseudonymiser is not None:
        result[DEIDENTIFIED_TEXT_COLUMN] = pseudonymised_texts_batch
    return result


def refactor_texts_and_find_mentions(
    dataset: Dataset,
    nlp_pipeline: Language,
    num_proc: int,
    pseudonymiser=None,
    seed: int | None = None,
) -> Dataset:
    """Process the dataset to structure notes and extract NYHA mentions."""
    # Refactor each original text to contain the special characters that will be
    # used for proper rendering in LabelStudio (i.e., addition of the non-breaking
    # space for the XML-like structure)
    dataset = dataset.map(
        lambda batch: {
            DEIDENTIFIED_TEXT_COLUMN: [
                structure_note(
                    text,
                )
                for text in batch[DEIDENTIFIED_TEXT_COLUMN]
            ]
        },
        load_from_cache_file=False,
        batched=True,
        batch_size=1000,
        num_proc=num_proc if num_proc > 1 else None,
    )
    # Find all the NYHAs and dates inside the notes by applying regular expressions
    fn_kwargs: dict = {"nlp_pipeline": nlp_pipeline}
    if pseudonymiser is not None:
        fn_kwargs["pseudonymiser"] = pseudonymiser
        fn_kwargs["seed"] = seed
    dataset = dataset.map(
        find_nyha_and_dates,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=1000,
        num_proc=num_proc if num_proc > 1 else None,
        remove_columns=[],
        load_from_cache_file=False,
    )

    return dataset


def build_nyha_mentions_map(
    file_path: str,
    data_folder_path: str,
    pseudonymiser=None,
    seed: int | None = None,
) -> Tuple[dict, list]:
    """
    This method loads from the given file path all the clinical notes and extracts
    the NYHA and date mentions, with their possible linkings and returns
    a single map containing all the extracted information, and a list of the
    note ids for which NYHA mentions were found.

    Args:
        file_path (str): The path to the file containing the clinical notes to process.
        data_folder_path (str): The path to the folder containing the data.

    Returns:
        Tuple[dict, list]: A map containing all the extracted information and a list of note
        IDs for which NYHA mentions were found.
    """
    # Load the dataset from the given file path and concatenate
    # the different text segments into a single note
    df = preprocess_notes(
        file_path,
        data_folder_path,
    )
    dataset = Dataset(df.to_arrow())
    # Select the number of workers to use for multiprocessing,
    # considering the SLURM environment variable if present, and the DEBUG mode
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_CPUS_PER_TASK))
    num_proc = min(16, num_workers, len(dataset))

    # Initialize the NLP pipeline that must find the NYHA and date mentions
    # by using regular expressions, refactor the texts to properly structure them
    # for the annotation tool, and find the NYHA and date mentions by applying the
    # pipeline to the dataset in a batched way
    nlp_pipeline = spacy.blank("nl")
    nlp_pipeline.add_pipe("nyha_component")
    dataset = refactor_texts_and_find_mentions(
        dataset, nlp_pipeline, num_proc, pseudonymiser=pseudonymiser, seed=seed
    )

    return build_final_note_map(dataset)
