# ── Dutch stopwords ───────────────────────────────────────────────────────────
# Primary: load from nltk (requires corpus to be downloaded).
# Fallback: bundled list covering the most common Dutch function words so the
#           pipeline works without network access.
_DUTCH_STOPWORDS_FALLBACK: frozenset[str] = frozenset(
    {
        "de",
        "het",
        "een",
        "en",
        "van",
        "in",
        "is",
        "dat",
        "op",
        "te",
        "voor",
        "met",
        "zijn",
        "er",
        "maar",
        "als",
        "aan",
        "ze",
        "ook",
        "uit",
        "bij",
        "hij",
        "we",
        "om",
        "dit",
        "niet",
        "nog",
        "of",
        "heeft",
        "worden",
        "door",
        "naar",
        "worden",
        "hebben",
        "was",
        "had",
        "worden",
        "hun",
        "hem",
        "haar",
        "zij",
        "mij",
        "ons",
        "u",
        "je",
        "jij",
        "mijn",
        "uw",
        "zijn",
        "onze",
        "jullie",
        "men",
        "zich",
        "al",
        "dan",
        "nu",
        "wel",
        "al",
        "zo",
        "meer",
        "reeds",
        "geen",
        "andere",
        "andere",
        "veel",
        "dit",
        "die",
        "deze",
        "welke",
        "wat",
        "wie",
        "hoe",
        "waar",
        "wanneer",
        "waarom",
        "toch",
        "dus",
        "want",
        "want",
        "want",
        "omdat",
        "terwijl",
        "hoewel",
        "nadat",
        "totdat",
        "zodra",
        "alhoewel",
        "zolang",
        "indien",
        "tenzij",
        "mits",
        "worden",
        "zal",
        "zou",
        "kan",
        "kon",
        "moet",
        "mocht",
        "mag",
        "mogen",
        "zouden",
        "zullen",
        "kunnen",
        "moeten",
    }
)


def load_dutch_stopwords() -> frozenset[str]:
    """
    Return the Dutch stopword set from nltk if available, otherwise the
    bundled fallback.  A warning is printed when falling back.
    """
    try:
        from nltk.corpus import stopwords

        words = frozenset(stopwords.words("dutch"))
        print("[text] Dutch stopwords loaded from nltk corpus.")
        return words
    except Exception:
        print(
            "[text] WARNING: nltk Dutch stopwords unavailable "
            "(run nltk.download('stopwords')); using bundled fallback list."
        )
        return _DUTCH_STOPWORDS_FALLBACK
