"""End-to-end orchestration of Tasks 1-5.

Every stage returns data; nothing prints. The notebook (or ``__main__`` below)
is responsible for presentation, which keeps the pipeline testable headlessly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import aggregation, exploration, filtering, transformation, visualization
from .dataset import generate_dataset, save_dataset
from .exploration import Exploration

logger = logging.getLogger(__name__)

DEFAULT_HIGH_RATING = 8.0
DEFAULT_RECENT_YEAR = 2010
DEFAULT_GENRE_SUBSET: tuple[str, ...] = ('Action', 'Comedy')


@dataclass
class TaskTwoResult:
    """Task 2 — one frame per filter."""

    recent: pd.DataFrame
    highly_rated: pd.DataFrame
    genre_subset: pd.DataFrame
    doubled_budget: pd.DataFrame
    by_named_directors: pd.DataFrame
    by_shared_first_name: pd.DataFrame


@dataclass
class TaskFourResult:
    """Task 4 — one object per aggregation."""

    rating_by_genre: pd.Series
    top_grossing_by_director: pd.DataFrame
    by_decade: pd.DataFrame
    by_country: pd.DataFrame
    profitable_pct_by_genre: pd.Series


@dataclass
class PipelineResult:
    """Everything the assignment produces."""

    dataset: pd.DataFrame
    transformed: pd.DataFrame
    task_one: Exploration
    task_two: TaskTwoResult
    task_four: TaskFourResult
    figures: list[Path] = field(default_factory=list)


def run_task_one(frame: pd.DataFrame) -> Exploration:
    """Shape, head, descriptive statistics and missing-value counts."""
    return exploration.explore(frame)


def run_task_two(
    frame: pd.DataFrame,
    recent_year: int = DEFAULT_RECENT_YEAR,
    high_rating: float = DEFAULT_HIGH_RATING,
    genres: tuple[str, ...] = DEFAULT_GENRE_SUBSET,
) -> TaskTwoResult:
    """Apply all five Task 2 filters.

    Both readings of Task 2.5 are returned. ``by_named_directors`` is the
    literal answer (empty on generated data, since the brief's dataset cannot
    contain Nolan or Spielberg); ``by_shared_first_name`` is the looser reading
    the classroom notebook silently substituted.
    """
    return TaskTwoResult(
        recent=filtering.released_after(frame, recent_year),
        highly_rated=filtering.rated_above(frame, high_rating),
        genre_subset=filtering.in_genres(frame, genres),
        doubled_budget=filtering.box_office_exceeds_budget_multiple(frame, 2.0),
        by_named_directors=filtering.by_directors(frame),
        by_shared_first_name=filtering.by_directors(frame, match_first_name_only=True),
    )


def run_task_three(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Profit, ROI, Length, Decade and RatingBand, returning a new frame."""
    return transformation.transform(frame)


def run_task_four(frame: pd.DataFrame) -> TaskFourResult:
    """Every Task 4 aggregation. Requires a frame from :func:`run_task_three`."""
    missing = set(transformation.DERIVED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f'Task 4 needs the Task 3 columns; missing {sorted(missing)}. '
            'Run run_task_three() first.'
        )

    return TaskFourResult(
        rating_by_genre=aggregation.average_rating_by_genre(frame),
        top_grossing_by_director=aggregation.highest_grossing_by_director(frame),
        by_decade=aggregation.budget_and_box_office_by_decade(frame),
        by_country=aggregation.statistics_by_country(frame),
        profitable_pct_by_genre=aggregation.profitable_percentage_by_genre(frame),
    )


def run_task_five(
    frame: pd.DataFrame,
    rating_by_genre: pd.Series,
    output_dir: Path | str = '.',
) -> list[Path]:
    """Draw all five figures into ``output_dir`` and return their paths."""
    output_dir = Path(output_dir)
    return [
        visualization.plot_average_rating_by_genre(
            rating_by_genre, output_dir / 'avg_rating_by_genre.png'),
        visualization.plot_budget_vs_box_office(
            frame, output_dir / 'budget_vs_boxoffice.png'),
        visualization.plot_runtime_distribution(
            frame, output_dir / 'runtime_distribution.png'),
        visualization.plot_roi_by_genre(
            frame, output_dir / 'roi_by_genre.png'),
        visualization.plot_yearly_trend(
            aggregation.average_by_year(frame), output_dir / 'budget_boxoffice_trend.png'),
    ]


def run(
    output_dir: Path | str = '.',
    n_movies: int = 200,
    seed: int = 42,
    dataset_path: Path | str | None = None,
    draw_figures: bool = True,
) -> PipelineResult:
    """Run the whole assignment end to end and return every artefact."""
    output_dir = Path(output_dir)

    logger.info('Generating %d movies (seed=%d)', n_movies, seed)
    frame = generate_dataset(n_movies=n_movies, seed=seed)
    if dataset_path is not None:
        save_dataset(frame, dataset_path)

    task_one = run_task_one(frame)
    task_two = run_task_two(frame)
    transformed = run_task_three(frame)
    task_four = run_task_four(transformed)
    figures = run_task_five(transformed, task_four.rating_by_genre, output_dir) if draw_figures else []

    return PipelineResult(
        dataset=frame,
        transformed=transformed,
        task_one=task_one,
        task_two=task_two,
        task_four=task_four,
        figures=figures,
    )


def main() -> None:
    """Print a readable summary of every task. Used by ``python -m``."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    result = run()

    print('\n=== Task 1: Basic Data Exploration ===')
    print(f'Shape: {result.task_one.n_rows} rows x {result.task_one.n_columns} columns')
    print(f'\nFirst 5 rows:\n{result.task_one.head}')
    print(f'\nDescriptive statistics:\n{result.task_one.describe}')
    print(f'\nMissing values:\n{result.task_one.missing_counts}')

    print('\n=== Task 2: Data Filtering and Selection ===')
    two = result.task_two
    print(f'Released after 2010:            {len(two.recent)}')
    print(f'Rated above 8.0:                {len(two.highly_rated)}')
    print(f'Action or Comedy:               {len(two.genre_subset)}')
    print(f'Box office > 2x budget:         {len(two.doubled_budget)}')
    print(f'By Nolan or Spielberg (exact):  {len(two.by_named_directors)}'
          '  <- the brief names directors this dataset cannot generate')
    print(f'Sharing those first names:      {len(two.by_shared_first_name)}')

    print('\n=== Task 3: Data Transformation ===')
    print(result.transformed[
        ['Title', 'Budget', 'BoxOffice', 'Profit', 'ROI', 'Length', 'Decade', 'RatingBand']
    ].head())

    print('\n=== Task 4: Aggregation and Grouping ===')
    four = result.task_four
    print(f'\nAverage rating by genre:\n{four.rating_by_genre}')
    print(f'\nHighest-grossing per director (top 5):\n{four.top_grossing_by_director.head()}')
    print(f'\nBudget and box office by decade:\n{four.by_decade}')
    print(f'\nStatistics by country:\n{four.by_country}')
    print(f'\nProfitable (ROI > 1) percentage by genre:\n{four.profitable_pct_by_genre}')

    print('\n=== Task 5: Data Visualization ===')
    for path in result.figures:
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
