"""Shared fixtures. Downloads NLTK corpora once for the whole session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the sibling package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_homework import preprocessing  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def nltk_resources():
    preprocessing.ensure_nltk_resources()


@pytest.fixture(scope='session')
def base_texts() -> list[str]:
    """Stand-in for the 20-newsgroups fetch, so tests need no network."""
    return [
        'The projector broke halfway through and nobody in the room complained.',
        'I spent the weekend rewiring an amplifier and reading old manuals.',
        'Sunday afternoon rain kept everyone indoors watching whatever was on.',
        'A long thread about compiler warnings that nobody managed to resolve.',
    ]


@pytest.fixture(scope='session')
def small_corpus(base_texts):
    from nlp_homework.dataset import generate_dataset
    return generate_dataset(n_reviews=20, seed=7, base_texts=base_texts)
