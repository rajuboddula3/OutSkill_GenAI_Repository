"""Task 1 — basic data exploration."""

from __future__ import annotations

import pytest

from pandas_homework import exploration


class TestExplore:

    def test_reports_the_frame_shape(self, raw_movies):
        result = exploration.explore(raw_movies)
        assert result.shape == (200, 9)
        assert result.n_rows == 200
        assert result.n_columns == 9

    def test_head_returns_the_requested_number_of_rows(self, raw_movies):
        assert len(exploration.explore(raw_movies, n_head=3).head) == 3

    def test_head_defaults_to_five(self, raw_movies):
        assert len(exploration.explore(raw_movies).head) == 5

    def test_head_handles_a_frame_shorter_than_the_request(self, tiny_movies):
        assert len(exploration.explore(tiny_movies, n_head=100).head) == len(tiny_movies)

    def test_describe_covers_only_numeric_columns(self, raw_movies):
        describe = exploration.explore(raw_movies).describe
        assert set(describe.columns) == {'Year', 'Runtime', 'Budget', 'BoxOffice', 'Rating'}

    def test_describe_count_excludes_missing_values(self, raw_movies):
        describe = exploration.explore(raw_movies).describe
        assert describe.loc['count', 'Rating'] == 190
        assert describe.loc['count', 'Year'] == 200

    def test_counts_missing_values_per_column(self, raw_movies):
        counts = exploration.explore(raw_movies).missing_counts
        assert counts['Rating'] == 10
        assert counts['Title'] == 0

    def test_lists_columns_that_have_missing_values(self, raw_movies):
        result = exploration.explore(raw_movies)
        assert set(result.columns_with_missing) == {
            'Runtime', 'Budget', 'BoxOffice', 'Rating'
        }

    def test_totals_missing_values(self, raw_movies):
        assert exploration.explore(raw_movies).total_missing == 40

    def test_reports_no_missing_values_for_a_complete_frame(self):
        from pandas_homework import dataset
        complete = dataset.generate_dataset(n_movies=30, missing_rate=0.0)
        result = exploration.explore(complete)
        assert result.columns_with_missing == []
        assert result.total_missing == 0

    def test_does_not_mutate_the_input(self, tiny_movies):
        before = tiny_movies.copy()
        exploration.explore(tiny_movies)
        assert tiny_movies.equals(before)

    def test_rejects_a_negative_head_size(self, tiny_movies):
        with pytest.raises(ValueError, match='non-negative'):
            exploration.explore(tiny_movies, n_head=-1)
