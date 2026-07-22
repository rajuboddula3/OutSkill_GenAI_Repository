"""Task 5 — figures are written, and figure handles are not leaked."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pandas_homework import aggregation, visualization


@pytest.fixture(autouse=True)
def no_leaked_figures():
    """Fail any test that leaves a matplotlib figure open.

    REGRESSION: the classroom notebook called plt.figure() five times and closed
    none, so a session accumulated handles until matplotlib warned about it.
    """
    plt.close('all')
    yield
    assert not plt.get_fignums(), f'figures left open: {plt.get_fignums()}'


def _draw_all(frame, output_dir):
    ratings = aggregation.average_rating_by_genre(frame)
    yearly = aggregation.average_by_year(frame)
    return [
        visualization.plot_average_rating_by_genre(ratings, output_dir / 'rating.png'),
        visualization.plot_budget_vs_box_office(frame, output_dir / 'scatter.png'),
        visualization.plot_runtime_distribution(frame, output_dir / 'runtime.png'),
        visualization.plot_roi_by_genre(frame, output_dir / 'roi.png'),
        visualization.plot_yearly_trend(yearly, output_dir / 'trend.png'),
    ]


class TestFigureOutput:

    def test_every_plot_writes_a_non_empty_png(self, transformed_movies, tmp_path):
        for path in _draw_all(transformed_movies, tmp_path):
            assert path.exists(), f'{path.name} was not written'
            assert path.stat().st_size > 1000, f'{path.name} looks empty'

    def test_every_plot_returns_the_path_it_wrote(self, transformed_movies, tmp_path):
        paths = _draw_all(transformed_movies, tmp_path)
        assert {p.parent for p in paths} == {tmp_path}

    def test_creates_missing_parent_directories(self, transformed_movies, tmp_path):
        target = tmp_path / 'deep' / 'nested' / 'rating.png'
        ratings = aggregation.average_rating_by_genre(transformed_movies)
        assert visualization.plot_average_rating_by_genre(ratings, target).exists()

    def test_accepts_a_string_path(self, transformed_movies, tmp_path):
        ratings = aggregation.average_rating_by_genre(transformed_movies)
        path = visualization.plot_average_rating_by_genre(
            ratings, str(tmp_path / 'rating.png'))
        assert path.exists()


class TestRobustness:

    def test_handles_data_with_missing_values(self, tiny_transformed, tmp_path):
        """tiny_transformed carries NaN in Runtime, Budget, BoxOffice and Rating."""
        for path in _draw_all(tiny_transformed, tmp_path):
            assert path.exists()

    def test_yearly_trend_handles_a_single_year(self, transformed_movies, tmp_path):
        one_year = transformed_movies[transformed_movies['Year'] == 2010]
        yearly = aggregation.average_by_year(one_year)
        assert visualization.plot_yearly_trend(yearly, tmp_path / 'trend.png').exists()

    def test_uses_a_headless_backend(self):
        """Importing the module must not require a display."""
        import matplotlib
        assert matplotlib.get_backend().lower() == 'agg'
