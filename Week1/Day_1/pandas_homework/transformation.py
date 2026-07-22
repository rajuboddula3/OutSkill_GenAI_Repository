"""Task 3 — derived columns.

All five derived columns are added by :func:`transform`, which returns a new
frame. The classroom notebook mutated ``movies_data`` in place across five
cells, so re-running a cell out of order silently changed later results.

The two banded columns are real ordered ``category`` dtypes, not object columns
holding strings. The brief asks to "convert to a categorical type"; an ordered
categorical also makes ``sort_values`` and ``groupby`` order bands correctly
instead of alphabetically ('Average' < 'Excellent' < 'Poor').
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Ordered bands for the 'Length' column (Task 3.3).
LENGTH_LABELS: tuple[str, ...] = ('Short', 'Medium', 'Long')

#: Ordered bands for the 'RatingBand' column (Task 3.5).
RATING_LABELS: tuple[str, ...] = ('Poor', 'Average', 'Excellent')

#: Columns added by :func:`transform`.
DERIVED_COLUMNS: tuple[str, ...] = (
    'Profit', 'ROI', 'Length', 'Decade', 'RatingBand',
)


def add_profit(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 'Profit' = BoxOffice - Budget (Task 3.1)."""
    return frame.assign(Profit=frame['BoxOffice'] - frame['Budget'])


def add_roi(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 'ROI' = (BoxOffice - Budget) / Budget (Task 3.2).

    A zero budget yields ``NaN`` rather than ``inf``. The specified generator
    never produces one (budgets are uniform over [5, 250]), but an infinite ROI
    would otherwise propagate silently into every Task 4 mean and Task 5 axis
    limit if this ran against real data.
    """
    budget = frame['Budget'].replace(0, np.nan)
    return frame.assign(ROI=(frame['BoxOffice'] - budget) / budget)


def add_length_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 'Length': Short (<90), Medium (90-120 inclusive), Long (>120).

    ``np.select`` rather than ``pd.cut``: the brief's bands are closed on both
    ends of the middle band, which no single ``right=`` setting expresses (
    ``right=False`` puts 120 in Long, ``right=True`` puts 90 in Short).
    """
    runtime = frame['Runtime']
    band = np.select(
        [runtime < 90, runtime <= 120],
        ['Short', 'Medium'],
        default='Long',
    )
    # np.select cannot express "leave NaN alone", so restore it afterwards.
    band = pd.Series(band, index=frame.index).where(runtime.notna())
    return frame.assign(
        Length=pd.Categorical(band, categories=LENGTH_LABELS, ordered=True)
    )


def add_decade(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 'Decade' as an ordered categorical such as '1990s' (Task 3.4).

    Ordered by the numeric decade it came from. The classroom version built a
    plain string column and relied on lexicographic ``sort_index``, which only
    happens to be correct while every year has four digits.
    """
    start = (frame['Year'] // 10) * 10
    labels = start.astype('Int64').astype(str) + 's'
    ordered = [f'{decade}s' for decade in sorted(start.dropna().unique())]
    return frame.assign(
        Decade=pd.Categorical(labels, categories=ordered, ordered=True)
    )


def add_rating_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 'RatingBand': Poor (0-4], Average (4-7], Excellent (7-10] (Task 3.5).

    Added as a new column rather than overwriting 'Rating': Task 4 still needs
    the numeric ratings to average, and Task 5 needs them as a colour scale.
    """
    band = pd.cut(
        frame['Rating'],
        bins=[-np.inf, 4, 7, np.inf],
        labels=list(RATING_LABELS),
        right=True,
    )
    return frame.assign(RatingBand=band)


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply all five Task 3 transformations, returning a new frame."""
    result = frame
    for step in (add_profit, add_roi, add_length_band, add_decade, add_rating_band):
        result = step(result)
    return result
