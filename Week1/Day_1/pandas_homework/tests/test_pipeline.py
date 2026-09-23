"""End-to-end orchestration of all five tasks."""

from __future__ import annotations

import pandas as pd
import pytest

from pandas_homework import pipeline, transformation


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    """One full pipeline run, shared across the module."""
    output_dir = tmp_path_factory.mktemp('figures')
    return pipeline.run(output_dir=output_dir, n_movies=200, seed=42)


class TestRun:

    def test_returns_the_generated_dataset(self, result):
        assert result.dataset.shape == (200, 9)

    def test_transformed_frame_carries_the_derived_columns(self, result):
        for column in transformation.DERIVED_COLUMNS:
            assert column in result.transformed.columns

    def test_leaves_the_raw_dataset_untransformed(self, result):
        """Task 3 must not reach back and mutate the Task 1 input."""
        assert 'Profit' not in result.dataset.columns

    def test_task_one_reports_the_right_shape(self, result):
        assert result.task_one.shape == (200, 9)
        assert result.task_one.total_missing == 40

    def test_task_four_produces_every_aggregation(self, result):
        four = result.task_four
        assert not four.rating_by_genre.empty
        assert not four.top_grossing_by_director.empty
        assert not four.by_decade.empty
        assert not four.by_country.empty
        assert not four.profitable_pct_by_genre.empty

    def test_writes_all_five_figures(self, result):
        assert len(result.figures) == 5
        for path in result.figures:
            assert path.exists() and path.stat().st_size > 1000

    def test_figures_land_in_the_requested_directory(self, result):
        assert len({path.parent for path in result.figures}) == 1

    def test_is_reproducible(self, tmp_path):
        a = pipeline.run(output_dir=tmp_path / 'a', n_movies=50, seed=7)
        b = pipeline.run(output_dir=tmp_path / 'b', n_movies=50, seed=7)
        assert a.dataset.equals(b.dataset)
        assert a.task_four.rating_by_genre.equals(b.task_four.rating_by_genre)

    def test_can_skip_figures(self, tmp_path):
        result = pipeline.run(output_dir=tmp_path, n_movies=50, draw_figures=False)
        assert result.figures == []
        assert not list(tmp_path.glob('*.png'))

    def test_can_save_the_dataset(self, tmp_path):
        target = tmp_path / 'movies.csv'
        pipeline.run(output_dir=tmp_path, n_movies=20, dataset_path=target,
                     draw_figures=False)
        assert target.exists()
        assert len(pd.read_csv(target)) == 20

    def test_accepts_a_string_output_dir(self, tmp_path):
        result = pipeline.run(output_dir=str(tmp_path), n_movies=20)
        assert len(result.figures) == 5


class TestTaskTwoResults:

    def test_filters_are_consistent_with_the_dataset(self, result):
        two = result.task_two
        assert (two.recent['Year'] > 2010).all()
        assert (two.highly_rated['Rating'] > 8.0).all()
        assert set(two.genre_subset['Genre']) == {'Action', 'Comedy'}

    def test_literal_director_question_returns_nothing(self, result):
        """The brief's two directors cannot occur in the brief's own dataset."""
        assert result.task_two.by_named_directors.empty

    def test_loose_reading_is_reported_separately(self, result):
        """Both readings are surfaced so the discrepancy is visible, rather
        than the looser one silently standing in for the literal answer."""
        assert not result.task_two.by_shared_first_name.empty
        assert set(result.task_two.by_shared_first_name['Director'].str.split().str[0]) == {
            'Christopher', 'Steven'
        }


class TestTaskFourGuard:

    def test_rejects_a_frame_without_the_task_three_columns(self, result):
        with pytest.raises(ValueError, match='Run run_task_three'):
            pipeline.run_task_four(result.dataset)

    def test_names_the_missing_columns(self, result):
        with pytest.raises(ValueError, match='ROI'):
            pipeline.run_task_four(result.dataset)

    def test_accepts_a_transformed_frame(self, result):
        assert pipeline.run_task_four(result.transformed) is not None


class TestStageFunctions:
    """Each stage must be usable on its own, not only through run()."""

    def test_task_one_runs_standalone(self, raw_movies):
        assert pipeline.run_task_one(raw_movies).n_rows == 200

    def test_task_two_runs_standalone(self, raw_movies):
        assert not pipeline.run_task_two(raw_movies).recent.empty

    def test_task_two_thresholds_are_configurable(self, raw_movies):
        two = pipeline.run_task_two(raw_movies, recent_year=2020, high_rating=9.0)
        assert (two.recent['Year'] > 2020).all()
        assert (two.highly_rated['Rating'] > 9.0).all()

    def test_task_three_runs_standalone(self, raw_movies):
        assert 'ROI' in pipeline.run_task_three(raw_movies).columns

    def test_task_five_runs_standalone(self, transformed_movies, tmp_path):
        from pandas_homework import aggregation
        ratings = aggregation.average_rating_by_genre(transformed_movies)
        paths = pipeline.run_task_five(transformed_movies, ratings, tmp_path)
        assert len(paths) == 5


class TestMain:

    def test_prints_a_summary_for_every_task(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pipeline.main()
        out = capsys.readouterr().out
        for task in ('Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'):
            assert task in out
