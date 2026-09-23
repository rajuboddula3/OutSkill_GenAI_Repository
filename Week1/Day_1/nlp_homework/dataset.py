"""Synthetic movie-review corpus generation (dataset section of the brief)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from .vocabulary import (
    ACTOR_NAMES,
    AWARD_NAMES,
    DIRECTOR_NAMES,
    MOVIE_TERMS,
    MOVIE_TITLES,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
)

DEFAULT_SEED = 42
DEFAULT_N_REVIEWS = 1000

_FALLBACK_WORDS = ['This', 'is', 'a', 'placeholder', 'review']


def load_base_texts(limit: int = 5000) -> list[str]:
    """Fetch newsgroup posts used as filler prose beneath the injected terms."""
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return [text for text in newsgroups.data[:limit] if text.strip()]


def generate_review(
    sentiment: int,
    base_texts: list[str],
    rng: random.Random,
    length_range: tuple[int, int] = (50, 500),
) -> str:
    """Build one synthetic review by seeding filler prose with domain terms.

    ``rng`` is injected rather than using the ``random`` module globals, so a
    generated corpus is reproducible regardless of what else in the process
    touches the global RNG.
    """
    words = rng.choice(base_texts).split() or list(_FALLBACK_WORDS)

    target_length = rng.randint(*length_range)
    words = words[:target_length]

    def insert(term: str) -> None:
        words.insert(rng.randint(0, len(words)), term)

    sentiment_pool = POSITIVE_WORDS if sentiment == 1 else NEGATIVE_WORDS
    for _ in range(rng.randint(3, 10)):
        insert(rng.choice(sentiment_pool))

    for _ in range(rng.randint(1, 5)):
        insert(rng.choice(MOVIE_TERMS))

    if rng.random() < 0.7:  # 70% of reviews carry named entities
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.6:
                insert(rng.choice(DIRECTOR_NAMES))
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.7:
                insert(rng.choice(ACTOR_NAMES))
        for _ in range(rng.randint(0, 2)):
            if rng.random() < 0.5:
                insert(rng.choice(MOVIE_TITLES))
        if rng.random() < 0.3:
            insert(rng.choice(AWARD_NAMES))

    return ' '.join(' '.join(words).split())


def generate_dataset(
    n_reviews: int = DEFAULT_N_REVIEWS,
    seed: int = DEFAULT_SEED,
    base_texts: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a balanced, shuffled, reproducible review corpus."""
    if n_reviews % 2:
        raise ValueError(f'n_reviews must be even for a balanced corpus, got {n_reviews}')

    rng = random.Random(seed)
    texts = base_texts if base_texts is not None else load_base_texts()
    if not texts:
        raise ValueError('base_texts is empty; cannot generate reviews')

    records = []
    for _ in range(n_reviews // 2):
        records.append({'review': generate_review(1, texts, rng), 'sentiment': 1})
        records.append({'review': generate_review(0, texts, rng), 'sentiment': 0})

    frame = pd.DataFrame(records)
    # Explicit random_state — the naive version's shuffle depended on numpy's
    # global RNG and so was not actually reproducible.
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {'review', 'sentiment'} - set(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing required column(s): {sorted(missing)}')
    return frame
