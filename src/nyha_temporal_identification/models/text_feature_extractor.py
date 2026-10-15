import re
import string

import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from nyha_temporal_identification.constants import NBSP
from nyha_temporal_identification.models.constants import (
    DEFAULT_WINDOW_SIZE,
    NYHA_PRED_END_INDEX,
    NYHA_PRED_START_INDEX,
    TEXT_COL,
)
from nyha_temporal_identification.models.dutch_stopwords import load_dutch_stopwords


class TextFeatureExtractor:
    """
    Self-contained text processing branch.

    Pipeline
    --------
    1. Window extraction  : slice text[max(0,start-W) : min(len,end+W)]
    2. Preprocessing      : lowercase → strip punctuation/digits → tokenise
                            → remove Dutch stopwords → (optional) stem
    3. Vectorisation      : CountVectorizer fit on train, transform val/test

    The fitted vectorizer is stored as self.vectorizer_ after fit(); it is
    then used to transform val and test with no re-fitting (no leakage).

    Parameters
    ----------
    window_size : int    Characters to include on each side of the span (default 200).
    apply_stemming : bool  Apply nltk SnowballStemmer("dutch") (default False).
    max_features : int | None  Passed to CountVectorizer to cap vocabulary size.

    Attributes
    ----------
    vectorizer_ : CountVectorizer   Fitted on training corpus (set after fit()).
    stopwords_  : frozenset[str]    Dutch stopword set actually used.
    vocab_size_ : int               Number of vocabulary terms after fitting.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        apply_stemming: bool = False,
        max_features: int | None = None,
    ) -> None:
        self.window_size = window_size
        self.apply_stemming = apply_stemming
        self.max_features = max_features

        self.stopwords_: frozenset[str] = load_dutch_stopwords()
        self.vectorizer_: CountVectorizer | None = None
        self.vocab_size_: int = 0

        if apply_stemming:
            try:
                from nltk.stem.snowball import SnowballStemmer

                self._stemmer = SnowballStemmer("dutch")
            except Exception as exc:
                raise ImportError(
                    "nltk SnowballStemmer unavailable. "
                    "Run nltk.download('punkt') or set apply_stemming=False."
                ) from exc
        else:
            self._stemmer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_window(self, text: str, start: int, end: int) -> str:
        """
        Return the substring of *text* in the window [start-W, end+W],
        clamped to document boundaries.

        Edge-case handling
        ──────────────────
        - If start > end (malformed indices), they are swapped silently.
        - Negative indices are clamped to 0.
        - Indices beyond len(text) are clamped to len(text).
        """
        if start > end:
            start, end = end, start
        n = len(text)
        lo = max(0, start - self.window_size)
        hi = min(n, end + self.window_size)
        return text[lo:hi]

    def _preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing chain to a single string:
          1. Lowercase
          2. Remove punctuation and digits
          3. Tokenise by whitespace
          4. Remove Dutch stopwords
          5. (Optional) Stem with SnowballStemmer("dutch")

        Returns the processed tokens joined by a single space, ready for
        CountVectorizer (which will tokenise again; this keeps the contract
        clean and allows CountVectorizer's own ngram/min_df logic to apply).
        """
        # Step 1 – lowercase
        text = text.lower()
        # Step 2 – remove punctuation and digits
        text = re.sub(r"[" + re.escape(string.punctuation) + r"\d]", " ", text)
        # Step 3 – tokenise by whitespace
        text = text.replace(NBSP, " ")
        tokens = text.split()
        # Step 4 – remove stopwords (and single-character tokens as noise)
        tokens = [t for t in tokens if t not in self.stopwords_ and len(t) > 1]
        # Step 5 – optional stemming
        if self._stemmer is not None:
            stemmed_tokens = []
            for t in tokens:
                stemmed_tokens.append(self._stemmer.stem(t))  # ty:ignore[invalid-argument-type, missing-argument]
            tokens = stemmed_tokens
        return " ".join(tokens)

    def _extract_and_preprocess(self, df: pl.DataFrame) -> list[str]:
        """
        Extract windowed text strings from a Polars DataFrame and preprocess
        each one.  Rows with null text or null span indices are replaced by an
        empty string (with a one-time warning if any are found).

        Parameters
        ----------
        df : pl.DataFrame   Must contain TEXT_COL, SPAN_START_COL, SPAN_END_COL.

        Returns
        -------
        list[str]   One preprocessed document string per row.
        """
        texts = df[TEXT_COL].to_list()
        starts = df[NYHA_PRED_START_INDEX].to_list()
        ends = df[NYHA_PRED_END_INDEX].to_list()

        null_rows = sum(
            1
            for t, s, e in zip(texts, starts, ends)
            if t is None or s is None or e is None
        )
        if null_rows:
            print(
                f"[text] WARNING: {null_rows} row(s) have null text/indices; "
                "replaced with empty string."
            )

        docs: list[str] = []
        for text, start, end in zip(texts, starts, ends):
            if text is None or start is None or end is None:
                docs.append("")
                continue
            window = self._extract_window(text, int(start), int(end))
            docs.append(self._preprocess(window))
        return docs

    def fit(self, train_df: pl.DataFrame) -> "TextFeatureExtractor":
        """
        Fit the CountVectorizer on the training corpus.

        Parameters
        ----------
        train_df : pl.DataFrame   Training split (must contain text columns).
        """
        docs = self._extract_and_preprocess(train_df)
        self.vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-z]{2,}\b",  # minimum 2-char tokens
        )
        self.vectorizer_.fit(docs)
        self.vocab_size_ = len(self.vectorizer_.vocabulary_)
        print(
            f"[text] Vectorizer fitted on {len(docs)} training documents; "
            f"vocabulary size = {self.vocab_size_:,} "
            f"(window={self.window_size}, stemming={self.apply_stemming})"
        )
        return self

    def transform(self, df: pl.DataFrame) -> sp.csr_matrix:
        """
        Transform a split into a sparse BoW feature matrix.

        Parameters
        ----------
        df : pl.DataFrame   Any split (train / val / test).

        Returns
        -------
        scipy.sparse.csr_matrix   Shape (n_rows, vocab_size).
        """
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before transform().")
        docs = self._extract_and_preprocess(df)
        return self.vectorizer_.transform(docs)  # pyright: ignore[reportReturnType]

    def fit_transform(self, train_df: pl.DataFrame) -> sp.csr_matrix:
        """Fit on train_df and return its sparse BoW matrix in one call."""
        self.fit(train_df)
        return self.transform(train_df)
