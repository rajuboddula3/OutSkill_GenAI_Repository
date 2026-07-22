"""Synthetic movie dataset generation (dataset section of the brief).

The brief's snippet seeds the *global* numpy RNG and then calls module-level
``np.random.*`` helpers. That makes the corpus reproducible only if nothing else
in the process has touched the global RNG in between. Here a ``RandomState`` is
threaded through explicitly instead. ``RandomState(seed)`` replays exactly the
same sequence as ``np.random.seed(seed)`` followed by module-level calls, so the
generated values still match the original classroom notebook column for column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_N_MOVIES = 200

#: Fraction of each nullable column blanked out, matching the brief.
DEFAULT_MISSING_RATE = 0.05

#: Columns the brief punches missing values into.
NULLABLE_COLUMNS: tuple[str, ...] = ('Runtime', 'Budget', 'BoxOffice', 'Rating')

GENRES: tuple[str, ...] = (
    'Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Thriller', 'Romance',
)

COUNTRIES: tuple[str, ...] = (
    'USA', 'UK', 'France', 'Japan', 'South Korea', 'India', 'Canada', 'Germany',
)

DIRECTOR_FIRST_NAMES: tuple[str, ...] = (
    'James', 'Steven', 'Christopher', 'Martin', 'Quentin',
    'David', 'Ridley', 'Sofia', 'Greta', 'Kathryn',
)

DIRECTOR_LAST_NAMES: tuple[str, ...] = (
    'Smith', 'Johnson', 'Williams', 'Jones', 'Brown',
    'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
)

#: Column order of a freshly generated frame, before any transformation.
RAW_COLUMNS: tuple[str, ...] = (
    'Title', 'Year', 'Genre', 'Runtime', 'Budget',
    'BoxOffice', 'Director', 'Rating', 'Country',
)


def generate_dataset(
    n_movies: int = DEFAULT_N_MOVIES,
    seed: int = DEFAULT_SEED,
    missing_rate: float = DEFAULT_MISSING_RATE,
) -> pd.DataFrame:
    """Generate the reproducible movie dataset described in ``pandas_homework.md``.

    Args:
        n_movies: Number of rows to generate.
        seed: Seed for the local ``RandomState``.
        missing_rate: Fraction of each column in :data:`NULLABLE_COLUMNS` set to
            ``NaN``, so Task 1's missing-value check has something to find.

    Raises:
        ValueError: If ``n_movies`` is not positive or ``missing_rate`` is
            outside ``[0, 1]``.
    """
    if n_movies <= 0:
        raise ValueError(f'n_movies must be positive, got {n_movies}')
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f'missing_rate must be within [0, 1], got {missing_rate}')

    rng = np.random.RandomState(seed)

    budgets = np.round(rng.uniform(5, 250, n_movies), 1)
    frame = pd.DataFrame({
        'Title': [f'Movie {i}' for i in range(1, n_movies + 1)],
        'Year': rng.randint(1990, 2023, n_movies),
        'Genre': rng.choice(GENRES, n_movies),
        'Runtime': rng.randint(75, 180, n_movies),
        'Budget': budgets,
        'BoxOffice': np.round(budgets * rng.uniform(0.5, 4, n_movies), 1),
        'Director': [
            f'{rng.choice(DIRECTOR_FIRST_NAMES)} {rng.choice(DIRECTOR_LAST_NAMES)}'
            for _ in range(n_movies)
        ],
        'Rating': np.round(rng.uniform(3, 9.5, n_movies), 1),
        'Country': rng.choice(COUNTRIES, n_movies),
    })

    n_missing = int(n_movies * missing_rate)
    for column in NULLABLE_COLUMNS:
        # Cast first: assigning NaN into an int column raises under pandas 3.
        frame[column] = frame[column].astype(float)
        blanked = rng.choice(n_movies, size=n_missing, replace=False)
        frame.loc[blanked, column] = np.nan

    return frame


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a movie CSV, failing loudly if the schema does not match.

    Task 1 of the brief allows loading from CSV rather than generating; this is
    that path, with the column check the brief's ``pd.read_csv`` call lacks.
    """
    frame = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing required column(s): {sorted(missing)}')
    return frame
