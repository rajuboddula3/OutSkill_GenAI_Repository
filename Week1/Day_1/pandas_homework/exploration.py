"""Task 1 — basic data exploration.

Every function returns data rather than printing it, so the results are
assertable in tests. Presentation is the notebook's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Exploration:
    """The full Task 1 answer set for one DataFrame."""

    head: pd.DataFrame
    shape: tuple[int, int]
    describe: pd.DataFrame
    missing_counts: pd.Series

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_columns(self) -> int:
        return self.shape[1]

    @property
    def columns_with_missing(self) -> list[str]:
        """Names of columns holding at least one missing value."""
        return self.missing_counts[self.missing_counts > 0].index.tolist()

    @property
    def total_missing(self) -> int:
        return int(self.missing_counts.sum())


def explore(frame: pd.DataFrame, n_head: int = 5) -> Exploration:
    """Answer all five Task 1 sub-questions in one pass.

    Raises:
        ValueError: If ``n_head`` is negative.
    """
    if n_head < 0:
        raise ValueError(f'n_head must be non-negative, got {n_head}')

    return Exploration(
        head=frame.head(n_head),
        shape=(frame.shape[0], frame.shape[1]),
        describe=frame.describe(),
        missing_counts=frame.isna().sum(),
    )
