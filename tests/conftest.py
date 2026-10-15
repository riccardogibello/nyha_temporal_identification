import pytest
import spacy

import nyha_temporal_identification.extraction.parsing  # noqa: F401 — registers nyha_component


@pytest.fixture(scope="session")
def nlp():
    pipeline = spacy.blank("nl")
    pipeline.add_pipe("nyha_component")
    return pipeline
