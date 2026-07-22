"""Shared fixtures for the pandas homework suite.

These tests live under the package rather than in ``Day_1/tests/`` because that
directory is the NLP homework suite: its ``conftest.py`` autouses an NLTK
download, and it already owns the names ``test_dataset``, ``test_exploration``
and ``test_pipeline``. Keeping the two suites apart means neither can slow down
or shadow the other. Run this one with ``pytest pandas_homework``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the package importable without an editable install. parents[2] is Day_1,
# the directory that contains pandas_homework/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pandas_homework import dataset, transformation  # noqa: E402


@pytest.fixture(scope='session')
def raw_movies() -> pd.DataFrame:
    """The default 200-row generated dataset, exactly as the brief specifies."""
    return dataset.generate_dataset()


@pytest.fixture(scope='session')
def transformed_movies(raw_movies: pd.DataFrame) -> pd.DataFrame:
    """``raw_movies`` with every Task 3 column added."""
    return transformation.transform(raw_movies)


@pytest.fixture
def tiny_movies() -> pd.DataFrame:
    """A small hand-built frame with known values, for exact assertions.

    Deliberately carries missing runtimes, budgets, box offices and ratings so
    NaN handling is exercised on every path rather than only on the randomly
    blanked generated data. Row 'D' is missing Budget, so its ROI is unknown —
    that row is what the Task 4.5 denominator regression turns on.
    """
    return pd.DataFrame({
        'Title': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Year': [1995, 2003, 2011, 2011, 2020, 1999],
        'Genre': ['Action', 'Action', 'Comedy', 'Comedy', 'Drama', 'Drama'],
        'Runtime': [80.0, 90.0, 120.0, 121.0, np.nan, 150.0],
        'Budget': [100.0, 50.0, 200.0, np.nan, 10.0, 80.0],
        'BoxOffice': [300.0, 40.0, 500.0, 100.0, np.nan, 160.0],
        'Director': [
            'Christopher Nolan', 'Christopher Brown', 'Steven Spielberg',
            'Steven Taylor', 'Greta Davis', 'Greta Davis',
        ],
        'Rating': [4.0, 7.0, 7.1, np.nan, 3.0, 9.0],
        'Country': ['USA', 'UK', 'USA', 'Japan', 'India', 'USA'],
    })


@pytest.fixture
def tiny_transformed(tiny_movies: pd.DataFrame) -> pd.DataFrame:
    """:func:`tiny_movies` with the Task 3 columns added."""
    return transformation.transform(tiny_movies)
