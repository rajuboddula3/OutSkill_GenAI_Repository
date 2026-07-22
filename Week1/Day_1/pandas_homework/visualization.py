"""Task 5 — figures.

Computation lives in :mod:`pandas_homework.aggregation`; this module only draws,
so every statistic stays testable without a display backend.

Each function takes an explicit output path and returns it, and every figure is
closed after being written. The classroom notebook called ``plt.figure()`` five
times without closing any, leaking figure handles and drawing a warning once
past twenty figures in a session.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def plot_average_rating_by_genre(ratings: pd.Series, output_path: Path) -> Path:
    """Bar chart of mean rating per genre (Task 5.1)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ratings.plot(kind='bar', color='skyblue', ax=ax)
    ax.set(title='Average Rating by Genre', xlabel='Genre', ylabel='Average Rating')
    ax.tick_params(axis='x', rotation=45)
    return _save(fig, output_path)


def plot_budget_vs_box_office(frame: pd.DataFrame, output_path: Path) -> Path:
    """Budget against box office, coloured by rating (Task 5.2)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    points = ax.scatter(
        frame['Budget'], frame['BoxOffice'],
        c=frame['Rating'], cmap='viridis', alpha=0.7, s=100,
    )
    fig.colorbar(points, ax=ax, label='Rating')
    ax.set(
        title='Budget vs. Box Office (coloured by Rating)',
        xlabel='Budget (millions USD)',
        ylabel='Box Office (millions USD)',
    )
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_runtime_distribution(frame: pd.DataFrame, output_path: Path, bins: int = 20) -> Path:
    """Histogram of runtimes (Task 5.3)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(frame['Runtime'].dropna(), bins=bins, color='purple', alpha=0.7)
    ax.set(
        title='Distribution of Movie Runtimes',
        xlabel='Runtime (minutes)',
        ylabel='Frequency',
    )
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_roi_by_genre(frame: pd.DataFrame, output_path: Path) -> Path:
    """Box plot of ROI spread per genre (Task 5.4)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Genre', y='ROI', data=frame, ax=ax)
    ax.set(
        title='Distribution of ROI by Genre',
        xlabel='Genre',
        ylabel='Return on Investment (ROI)',
    )
    ax.tick_params(axis='x', rotation=45)
    return _save(fig, output_path)


def plot_yearly_trend(yearly: pd.DataFrame, output_path: Path) -> Path:
    """Mean budget and box office over time (Task 5.5)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(yearly.index, yearly['Budget'], marker='o', color='blue', label='Avg Budget')
    ax.plot(yearly.index, yearly['BoxOffice'], marker='s', color='green', label='Avg Box Office')
    span = f'{int(yearly.index.min())}-{int(yearly.index.max())}' if len(yearly) else 'no data'
    ax.set(
        title=f'Average Budget and Box Office by Year ({span})',
        xlabel='Year',
        ylabel='Amount (millions USD)',
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def _save(fig, output_path: Path) -> Path:
    """Write a figure and close it, so long runs do not leak figure handles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path
