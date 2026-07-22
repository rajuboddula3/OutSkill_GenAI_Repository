"""Task 4 — aggregation and grouping.

Two notes that apply throughout:

* Every ``groupby`` over a categorical key passes ``observed=True``. The default
  flipped in pandas 3 and emits a ``FutureWarning`` in 2.2; without it, grouping
  by 'Decade' also yields rows for decades that no movie occupies.
* Aggregations are expressed on a selected column (``groupby(k)[col]``) rather
  than via ``groupby(k).apply(func)``. Applying a function to the whole group
  frame triggers ``DeprecationWarning: DataFrameGroupBy.apply operated on the
  grouping columns`` on pandas 2.2 and changes behaviour in pandas 3.
"""

from __future__ import annotations

import pandas as pd


def average_rating_by_genre(frame: pd.DataFrame) -> pd.Series:
    """Mean rating per genre, best first (Task 4.1).

    Missing ratings are skipped by ``mean``, so a genre is averaged over the
    movies that actually have a rating.
    """
    return (
        frame.groupby('Genre', observed=True)['Rating']
        .mean()
        .sort_values(ascending=False)
    )


def highest_grossing_by_director(frame: pd.DataFrame) -> pd.DataFrame:
    """The single top-grossing movie for each director (Task 4.2).

    Directors whose every movie is missing a box office figure are absent from
    the result — ``idxmax`` over an all-NaN group has no answer to give.
    """
    known = frame.dropna(subset=['BoxOffice'])
    if known.empty:
        return frame.loc[[], ['Director', 'Title', 'BoxOffice']]

    top_indices = known.groupby('Director', observed=True)['BoxOffice'].idxmax()
    return (
        known.loc[top_indices, ['Director', 'Title', 'BoxOffice']]
        .sort_values('BoxOffice', ascending=False)
    )


def budget_and_box_office_by_decade(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean budget and box office per decade, in chronological order (Task 4.3).

    'Decade' is an ordered categorical, so ``sort_index`` is chronological
    rather than lexicographic.
    """
    return (
        frame.groupby('Decade', observed=True)[['Budget', 'BoxOffice']]
        .mean()
        .sort_index()
    )


def statistics_by_country(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean rating, total budget and total box office per country (Task 4.4)."""
    return (
        frame.groupby('Country', observed=True)
        .agg(
            MeanRating=('Rating', 'mean'),
            TotalBudget=('Budget', 'sum'),
            TotalBoxOffice=('BoxOffice', 'sum'),
        )
        .sort_values('TotalBoxOffice', ascending=False)
    )


def profitable_percentage_by_genre(
    frame: pd.DataFrame, threshold: float = 1.0
) -> pd.Series:
    """Percent of each genre's movies with ROI above ``threshold`` (Task 4.5).

    The denominator counts only movies whose ROI is *known*. Dividing by the
    full group size — as the classroom version did — charges every movie with a
    missing budget or box office against the genre as if it were unprofitable,
    which understates every percentage. With 5% missing in each of two columns,
    that is a systematic error of roughly 10%.

    Genres where no movie has a known ROI are reported as ``NaN``, not 0: "we
    cannot tell" and "none were profitable" are different answers.
    """
    def percentage(roi: pd.Series) -> float:
        known = roi.dropna()
        if known.empty:
            return float('nan')
        return float((known > threshold).mean() * 100)

    return (
        frame.groupby('Genre', observed=True)['ROI']
        .apply(percentage)
        .sort_values(ascending=False)
    )


def average_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean budget and box office per release year, for the Task 5.5 trend line."""
    return (
        frame.groupby('Year', observed=True)[['Budget', 'BoxOffice']]
        .mean()
        .sort_index()
    )
