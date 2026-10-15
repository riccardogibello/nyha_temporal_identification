import csv
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass, field

from nyha_temporal_identification.constants import (
    DUTCH_FIRST_NAMES_FILE_NAME,
    DUTCH_HOSPITALS_FILE_NAME,
    DUTCH_PLACES_FILE_NAME,
    DUTCH_SURNAMES_FILE_NAME,
    PSEUDONYMISATION_FOLDER_NAME,
    RESOURCES_FOLDER_NAME,
)

_log = logging.getLogger(__name__)

DEANON_TOKEN_RE = re.compile(r"\[([A-Z][A-Z_]*)(?:-(\d+))?\]")

DUTCH_TUSSENVOEGSELS = [
    "de", "van", "van der", "van den", "van de", "ter", "ten", "op den",
]

CATEGORY_ALIASES: dict[str, str] = {
    # person
    "PERSON": "person", "PERSOON": "person", "PATIENT": "person",
    "PATIENT_NAME": "person", "DOCTOR": "person", "ARTS": "person",
    "NAME": "person", "NAAM": "person", "INITIAAL": "person", "INITIALEN": "person",
    # institution
    "HOSPITAL": "institution", "ZIEKENHUIS": "institution", "INSTELLING": "institution",
    "INSTITUTION": "institution", "ORGANIZATION": "institution", "ORGANISATIE": "institution",
    "ORGANISATION": "institution", "ZORGINSTELLING": "institution",
    # place
    "LOCATION": "place", "LOCATIE": "place", "PLAATS": "place", "PLACE": "place",
    "CITY": "place", "STAD": "place", "ADDRESS": "place", "ADRES": "place",
    "STREET": "place", "STRAAT": "place", "GEMEENTE": "place",
    # date
    "DATE": "date", "DATUM": "date",
    # age
    "AGE": "age", "LEEFTIJD": "age",
    # phone
    "PHONE": "phone", "TELEFOON": "phone", "TELEFOONNUMMER": "phone",
    "TEL": "phone", "FAX": "phone", "FAXNUMMER": "phone",
    # email
    "EMAIL": "email", "MAIL": "email", "MAILADRES": "email", "EMAILADRES": "email",
    # id
    "ID": "id", "BSN": "id", "PATIENTNUMMER": "id", "PATIENT_NUMBER": "id",
    "NUMBER": "id", "NUMMER": "id", "MRN": "id", "ZIS": "id", "IDENTIFIER": "id",
    # url
    "URL": "url", "WEBSITE": "url", "WEBADRES": "url",
    # number word
    "NUMMERWOORD": "nummerwoord",
}

_EMAIL_FIRST = ["anna", "jan", "petra", "kees", "marie", "piet", "lisa", "henk",
                "sanne", "tom", "inge", "bas", "eva", "rob", "lotte", "erik"]
_EMAIL_LAST = ["jansen", "smit", "bakker", "de vries", "mulder", "visser",
               "berg", "meijer", "boer", "dijkstra", "bos", "vos", "graaf"]


# ---------------------------------------------------------------------------
# Data-backed generators
# ---------------------------------------------------------------------------

class PersonNameGenerator:
    def __init__(
        self,
        first_names: list[tuple[str, str, int]],
        surnames: list[tuple[str, int]],
        tussenvoegsels: list[str] = DUTCH_TUSSENVOEGSELS,
    ):
        self._first_names = [n for n, _, _ in first_names]
        self._first_weights = [max(w, 1) for _, _, w in first_names]
        self._surnames = [n for n, _ in surnames]
        self._surname_weights = [max(w, 1) for _, w in surnames]
        self._tussenvoegsels = tussenvoegsels

    def __call__(self, rng: random.Random) -> str:
        first = rng.choices(self._first_names, weights=self._first_weights, k=1)[0]
        surname = rng.choices(self._surnames, weights=self._surname_weights, k=1)[0]
        if rng.random() < 0.25:
            tv = rng.choice(self._tussenvoegsels)
            return f"{first} {tv} {surname}"
        return f"{first} {surname}"


class InstitutionNameGenerator:
    def __init__(self, hospitals: list[str]):
        self._hospitals = hospitals

    def __call__(self, rng: random.Random) -> str:
        return rng.choice(self._hospitals)


class PlaceNameGenerator:
    def __init__(self, places: list[tuple[str, int]]):
        self._places = [n for n, _ in places]
        self._weights = [max(w, 1) for _, w in places]

    def __call__(self, rng: random.Random) -> str:
        return rng.choices(self._places, weights=self._weights, k=1)[0]


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def random_date(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    year = rng.randint(1990, 2025)
    return f"{day:02d}-{month:02d}-{year}"


def random_age(rng: random.Random) -> str:
    return str(rng.randint(1, 100))


def random_phone(rng: random.Random) -> str:
    if rng.random() < 0.5:
        digits = "".join(str(rng.randint(0, 9)) for _ in range(8))
        return f"06-{digits}"
    area = rng.randint(10, 99)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return f"0{area}-{digits}"


def random_email(rng: random.Random) -> str:
    first = rng.choice(_EMAIL_FIRST)
    last = rng.choice(_EMAIL_LAST).replace(" ", "")
    return f"{first}.{last}@example.nl"


def random_id(rng: random.Random) -> str:
    length = rng.randint(7, 10)
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def random_url(rng: random.Random) -> str:
    path = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=rng.randint(4, 10)))
    return f"https://www.example.nl/{path}"


def generic(rng: random.Random) -> str:
    return "Onbekend"


_DUTCH_NUMBER_WORDS = [
    "één", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien",
    "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien", "zeventien",
    "achttien", "negentien", "twintig",
    "eerste", "tweede", "derde", "vierde", "vijfde", "zesde", "zevende", "achtste",
    "negende", "tiende",
]


def random_number_word(rng: random.Random) -> str:
    return rng.choice(_DUTCH_NUMBER_WORDS)


PASSTHROUGH_CATEGORIES: frozenset[str] = frozenset({
    # Dutch workflow/status labels — not PHI
    "AFGEROND", "VERLOPEN", "GEANNULEERD", "WAARSCHUWING",
    "STANDBY", "NIEUW", "NEG", "LEVENSLANG",
    # Medical and clinical abbreviations — not PHI
    "NYHA", "HF", "TIA", "CRP", "CT", "ESR", "EMDR", "MRA",
    "NRS", "IFX", "FMD", "RR", "THC", "TCGA", "ADPKD",
})


# ---------------------------------------------------------------------------
# Replacement / PseudonymisationResult
# ---------------------------------------------------------------------------

@dataclass
class Replacement:
    raw_start: int
    raw_end: int
    new_start: int
    new_end: int
    raw_category: str
    canonical_category: str
    index: str | None
    original_token: str
    replacement: str


@dataclass
class PseudonymisationResult:
    new_text: str
    replacements: list[Replacement]
    mapping: dict[tuple[str, str | None], str]

    def remap(self, orig_start: int, orig_end: int) -> tuple[int, int]:
        new_start = orig_start
        new_end = orig_end
        for r in self.replacements:
            len_diff = len(r.replacement) - len(r.original_token)
            if r.raw_end <= orig_start:
                new_start += len_diff
                new_end += len_diff
            elif r.raw_end <= orig_end:
                new_end += len_diff
        return new_start, new_end


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _load_first_names(path: str) -> list[tuple[str, str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            gender = row.get("gender", "x").strip()
            try:
                weight = int(float(row.get("weight", "1") or "1"))
            except (ValueError, TypeError):
                weight = 1
            if name:
                rows.append((name, gender, max(weight, 1)))
    return rows


def _load_surnames(path: str) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            try:
                weight = int(float(row.get("weight", "1") or "1"))
            except (ValueError, TypeError):
                weight = 1
            if name:
                rows.append((name, max(weight, 1)))
    return rows


def _load_places(path: str) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            try:
                weight = int(float(row.get("weight", "1") or "1"))
            except (ValueError, TypeError):
                weight = 1
            if name:
                rows.append((name, max(weight, 1)))
    return rows


def _load_hospitals(path: str) -> list[str]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            if name:
                rows.append(name)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_note_rng(seed: int, note_id) -> random.Random:
    digest = hashlib.md5(f"{seed}:{note_id}".encode()).digest()
    stable_seed = int.from_bytes(digest[:8], "big")
    return random.Random(stable_seed)


def get_pseudonymisation_data_dir(override: str | None = None) -> str:
    if override is not None:
        return override
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            RESOURCES_FOLDER_NAME,
            PSEUDONYMISATION_FOLDER_NAME,
        )
    )


# ---------------------------------------------------------------------------
# Pseudonymiser
# ---------------------------------------------------------------------------

class Pseudonymiser:
    def __init__(
        self,
        data_dir: str | None = None,
        aliases: dict[str, str] = CATEGORY_ALIASES,
        fallback: str = "generic",
        token_re: re.Pattern = DEANON_TOKEN_RE,
        passthrough: frozenset[str] = PASSTHROUGH_CATEGORIES,
    ):
        self._aliases = aliases
        self._fallback = fallback
        self._token_re = token_re
        self._passthrough = passthrough
        self._warned_categories: set[str] = set()

        resolved_dir = get_pseudonymisation_data_dir(data_dir)
        if not os.path.isdir(resolved_dir):
            raise FileNotFoundError(
                f"Pseudonymisation data directory not found: {resolved_dir}\n"
                "Build it with: python scripts/build_pseudonymisation_data.py --data-dir <data_folder>"
            )
        first_names = _load_first_names(os.path.join(resolved_dir, DUTCH_FIRST_NAMES_FILE_NAME))
        surnames = _load_surnames(os.path.join(resolved_dir, DUTCH_SURNAMES_FILE_NAME))
        places = _load_places(os.path.join(resolved_dir, DUTCH_PLACES_FILE_NAME))
        hospitals = _load_hospitals(os.path.join(resolved_dir, DUTCH_HOSPITALS_FILE_NAME))

        self._generators: dict[str, object] = {
            "person": PersonNameGenerator(first_names, surnames),
            "institution": InstitutionNameGenerator(hospitals),
            "place": PlaceNameGenerator(places),
            "date": random_date,
            "age": random_age,
            "phone": random_phone,
            "email": random_email,
            "id": random_id,
            "url": random_url,
            "nummerwoord": random_number_word,
            "generic": generic,
        }

    def _resolve_generator(self, raw_category: str) -> tuple[str, object]:
        canonical = self._aliases.get(raw_category)
        if canonical is None:
            if raw_category not in self._warned_categories:
                _log.warning(
                    "Pseudonymiser: unknown deduce category %r — using fallback generator",
                    raw_category,
                )
                self._warned_categories.add(raw_category)
            canonical = self._fallback
        return canonical, self._generators[canonical]

    def pseudonymise(self, text: str, rng: random.Random) -> PseudonymisationResult:
        cache: dict[tuple[str, str | None], str] = {}
        replacements: list[Replacement] = []
        parts: list[str] = []
        cursor = 0
        offset_delta = 0

        for m in self._token_re.finditer(text):
            raw_category = m.group(1)
            index = m.group(2)

            if raw_category in self._passthrough:
                parts.append(text[cursor:m.end()])
                cursor = m.end()
                continue

            cache_key = (raw_category, index)

            if cache_key not in cache:
                canonical, generator = self._resolve_generator(raw_category)
                cache[cache_key] = generator(rng)  # type: ignore[operator]
                # record canonical for later
                cache_key_canonical = canonical
            else:
                canonical, _ = self._resolve_generator(raw_category)
                cache_key_canonical = canonical

            value = cache[cache_key]

            raw_start = m.start()
            raw_end = m.end()
            new_start = raw_start + offset_delta
            new_end = new_start + len(value)

            parts.append(text[cursor:raw_start])
            parts.append(value)
            cursor = raw_end

            replacements.append(
                Replacement(
                    raw_start=raw_start,
                    raw_end=raw_end,
                    new_start=new_start,
                    new_end=new_end,
                    raw_category=raw_category,
                    canonical_category=canonical,
                    index=index,
                    original_token=m.group(0),
                    replacement=value,
                )
            )
            offset_delta += len(value) - (raw_end - raw_start)

        parts.append(text[cursor:])
        new_text = "".join(parts)

        return PseudonymisationResult(
            new_text=new_text,
            replacements=replacements,
            mapping=cache,
        )
