"""Task 2 — data filtering and selection.

Each filter returns a new frame; none mutate the input. Comparisons against
``NaN`` are always ``False`` in pandas, so rows with missing values are dropped
by the numeric filters — that is the intended behaviour here and is asserted in
the tests rather than left implicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

#: The directors named in Task 2.5 of the brief.
BRIEF_DIRECTORS: tuple[str, ...] = ('Christopher Nolan', 'Steven Spielberg')


def released_after(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Movies released strictly after ``year`` (Task 2.1)."""
    return frame[frame['Year'] > year]


def rated_above(frame: pd.DataFrame, rating: float) -> pd.DataFrame:
    """Movies rated strictly above ``rating`` (Task 2.2).

    Rows with a missing rating are excluded.
    """
    return frame[frame['Rating'] > rating]


def in_genres(frame: pd.DataFrame, genres: Iterable[str]) -> pd.DataFrame:
    """Movies whose genre is any of ``genres`` (Task 2.3).

    ``isin`` scales to any number of genres, unlike chaining ``|`` per genre.
    """
    return frame[frame['Genre'].isin(list(genres))]


def box_office_exceeds_budget_multiple(
    frame: pd.DataFrame, multiple: float = 2.0
) -> pd.DataFrame:
    """Movies whose box office beat ``multiple`` times their budget (Task 2.4).

    Rows missing either figure are excluded, since the comparison is undefined.
    """
    return frame[frame['BoxOffice'] > multiple * frame['Budget']]


def by_directors(
    frame: pd.DataFrame,
    directors: Sequence[str] = BRIEF_DIRECTORS,
    *,
    match_first_name_only: bool = False,
) -> pd.DataFrame:
    """Movies by the given directors (Task 2.5).

    The brief asks for Christopher Nolan and Steven Spielberg, but the dataset
    it specifies builds director names by pairing a random first name with a
    random last name — neither director can ever occur. Exact matching is
    therefore correct *and* returns nothing on the generated data.

    ``match_first_name_only=True`` reproduces the looser interpretation (every
    Christopher and every Steven). It is opt-in and separately named because it
    answers a materially different question: on the default 200-row dataset it
    matches roughly 40 movies by ~20 distinct directors, not 2.

    Args:
        frame: Source movies.
        directors: Full names to match.
        match_first_name_only: Match on the first token of each name instead of
            the whole name.
    """
    names = list(directors)
    if not names:
        return frame.iloc[:0]

    if not match_first_name_only:
        return frame[frame['Director'].isin(names)]

    first_names = {name.split()[0] for name in names if name.split()}
    # Split-and-compare rather than str.contains: 'Christopher' as a regex would
    # also match a surname containing it, and would match mid-token.
    return frame[frame['Director'].str.split().str[0].isin(first_names)]
