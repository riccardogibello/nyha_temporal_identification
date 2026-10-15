import datetime
import logging
import re
from typing import Optional

import spacy

from nyha_temporal_identification.data_model.general import Mentions, NyhaMention
from nyha_temporal_identification.extraction.parsing import find_nyha_class

CURRENT_ANCHORS = [
    # --- English ---
    r"\bcurrently\b",
    r"\bpresently\b",
    r"\bat this time\b",
    r"\bnow\b",
    r"\bat present\b",
    r"\btoday\b",
    r"\bactive\b",
    r"\bthis visit\b",
    r"\bon current\b",
    r"\bremains\b",
    # --- Dutch: direct temporal markers ---
    r"\bmomenteel\b",  # currently / at the moment
    r"\bop dit moment\b",  # at this moment
    r"\bop heden\b",  # as of today
    r"\bheden\b",  # today / present (formal/clinical)
    r"\bnu\b",  # now
    r"\bthans\b",  # currently (formal)
    r"\btegenwoordig\b",  # nowadays / currently
    r"\bvandaag\b",  # today
    r"\bdit bezoek\b",  # this visit
    r"\bhuidige\b",  # current (adjective: 'huidige klasse')
    r"\bactief\b",  # active
    # --- Dutch: clinical functional status markers ---
    r"\bbehoudt\b",  # retains / remains (3rd person: 'patiënt behoudt NYHA II')
    r"\bblijft\b",  # stays / remains
    r"\bis geclassificeerd\b",  # is classified as
    r"\bverdraagt\b",  # tolerates
    r"\bpresteert\b",  # performs (functional capacity)
    r"\bkan\b",  # can (ability marker: 'kan trappen lopen')
]

HISTORICAL_ANCHORS = [
    # --- English ---
    r"\bpreviously\b",
    r"\bprior\b",
    r"\bformerly\b",
    r"\bhistorically\b",
    r"\bwas\b",
    r"\bbefore\b",
    r"\blast\s+\w+\b",
    r"\bimproved from\b",
    r"\breduced from\b",
    r"\bworsened from\b",
    # --- Dutch: direct temporal markers ---
    r"\bvroeger\b",  # previously / in the past
    r"\beerder\b",  # earlier / previously
    r"\bvoorheen\b",  # formerly / previously (clinical standard term)
    r"\bin het verleden\b",  # in the past
    r"\bvorig\w*\b",  # vorig / vorige / vorigen (previous, last)
    r"\bvoor\b",  # before (context-dependent — use with caution)
    r"\bdestijds\b",  # at that time
    r"\btoentertijd\b",  # at that time (more emphatic)
    r"\bindertijd\b",  # at one time / back then
    # --- Dutch: improvement / worsening transitions ---
    r"\bverbeterd van\b",  # improved from
    r"\bverslechterd van\b",  # worsened from
    r"\bafgenomen van\b",  # decreased from
    r"\btoegenomen van\b",  # increased from
    r"\bgedaald van\b",  # dropped from
    r"\bgestegen van\b",  # risen from
    r"\bvan\s+NYHA\b",  # 'van NYHA III naar II' (from NYHA ... to ...)
    # --- Dutch: past tense auxiliary verbs (strong historical signal) ---
    r"\bwas\b",  # was (also in English list)
    r"\bwerd\b",  # became / was (passive past)
    r"\bwerden\b",  # were (passive past plural)
    r"\bhad\b",  # had
    r"\bhadden\b",  # had (plural)
    r"\bis geweest\b",  # has been (perfect)
    r"\bwas geclassificeerd\b",  # was classified as
    # --- Dutch: pre-intervention markers (common in cardiology notes) ---
    r"\bvóór\s+(de\s+)?(operatie|ingreep|TAVI|ablatie|bypass|transplantatie)\b",
    r"\bpre[- ]?(operatief|interventie|procedureel)\b",
    r"\bvoor\s+(de\s+)?(operatie|ingreep|behandeling)\b",
]

CURRENT_RE = re.compile("|".join(CURRENT_ANCHORS), re.IGNORECASE)
HISTORICAL_RE = re.compile("|".join(HISTORICAL_ANCHORS), re.IGNORECASE)


CLASSIFICATION_CONTEXT_WINDOW: int = 1000  # chars on each side of the mention

try:
    import spacy  # type: ignore[no-redef]

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Date-linkage enrichment will be skipped.")

_nlp_pipeline: Optional[spacy.language.Language] = None


def get_nlp_pipeline() -> Optional[spacy.language.Language]:
    """Initialise and return the spaCy pipeline (singleton)."""
    global _nlp_pipeline
    if _nlp_pipeline is None and _SPACY_AVAILABLE:
        _nlp_pipeline = spacy.blank("nl")
        _nlp_pipeline.add_pipe("nyha_component")
    return _nlp_pipeline


def spacy_preprocess(
    deidentified_text: str,
    authored_datetime: Optional[str] = None,
) -> list[NyhaMention]:
    """
    Run the spaCy nyha_component on the note and return extracted mentions.

    Each mention includes the span text, NYHA class, nearest linked date,
    and character distance to that date.
    """
    nlp = get_nlp_pipeline()
    if nlp is None:
        return []

    authored_dt: Optional[datetime.datetime] = None
    if authored_datetime:
        try:
            authored_dt = datetime.datetime.fromisoformat(authored_datetime)
        except ValueError:
            logging.warning(
                f"Could not parse authored_datetime '{authored_datetime}'; "
                "proceeding without it."
            )

    try:
        result: Mentions = find_nyha_class(
            text=deidentified_text,
            authored_datetime=authored_dt,
            nlp_pipeline=nlp,
        )
        return result.extracted_nyha_mentions
    except Exception as exc:
        logging.warning(f"spaCy pre-processing failed: {exc}")
        return []
