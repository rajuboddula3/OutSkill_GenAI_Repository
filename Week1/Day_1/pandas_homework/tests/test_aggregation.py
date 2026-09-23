"""Task 4 — aggregation and grouping, including the two arithmetic regressions."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from pandas_homework import aggregation


@contextmanager
def warnings_as_errors():
    """Turn Future/Deprecation warnings into failures for the enclosed block."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        warnings.simplefilter('error', DeprecationWarning)
        yield


class TestAverageRatingByGenre:

    def test_averages_per_genre(self, tiny_transformed):
        result = aggregation.average_rating_by_genre(tiny_transformed)
        assert result['Action'] == pytest.approx(5.5)   # (4.0 + 7.0) / 2
        assert result['Drama'] == pytest.approx(6.0)    # (3.0 + 9.0) / 2

    def test_skips_missing_ratings(self, tiny_transformed):
        """Comedy holds 7.1 and a NaN; the mean is over the known value only."""
        result = aggregation.average_rating_by_genre(tiny_transformed)
        assert result['Comedy'] == pytest.approx(7.1)

    def test_is_sorted_best_first(self, transformed_movies):
        result = aggregation.average_rating_by_genre(transformed_movies)
        assert result.is_monotonic_decreasing

    def test_covers_every_genre(self, transformed_movies):
        result = aggregation.average_rating_by_genre(transformed_movies)
        assert set(result.index) == set(transformed_movies['Genre'].unique())


class TestHighestGrossingByDirector:

    def test_picks_the_top_film_per_director(self, tiny_transformed):
        result = aggregation.highest_grossing_by_director(tiny_transformed)
        top = dict(zip(result['Director'], result['Title']))
        assert top['Greta Davis'] == 'F'  # 160 beats E's missing value

    def test_has_one_row_per_director(self, tiny_transformed):
        result = aggregation.highest_grossing_by_director(tiny_transformed)
        assert not result['Director'].duplicated().any()

    def test_excludes_directors_with_no_known_box_office(self):
        frame = pd.DataFrame({
            'Director': ['Known', 'Unknown', 'Unknown'],
            'Title': ['a', 'b', 'c'],
            'BoxOffice': [10.0, np.nan, np.nan],
        })
        result = aggregation.highest_grossing_by_director(frame)
        assert list(result['Director']) == ['Known']

    def test_returns_an_empty_frame_when_nothing_is_known(self):
        frame = pd.DataFrame({
            'Director': ['a', 'b'], 'Title': ['x', 'y'], 'BoxOffice': [np.nan, np.nan],
        })
        result = aggregation.highest_grossing_by_director(frame)
        assert result.empty
        assert list(result.columns) == ['Director', 'Title', 'BoxOffice']

    def test_selects_the_actual_maximum(self, transformed_movies):
        result = aggregation.highest_grossing_by_director(transformed_movies)
        known = transformed_movies.dropna(subset=['BoxOffice'])
        expected = known.groupby('Director')['BoxOffice'].max()
        for _, row in result.iterrows():
            assert row['BoxOffice'] == expected[row['Director']]


class TestByDecade:

    def test_averages_budget_and_box_office(self, tiny_transformed):
        result = aggregation.budget_and_box_office_by_decade(tiny_transformed)
        # 1990s holds A (budget 100) and F (budget 80).
        assert result.loc['1990s', 'Budget'] == pytest.approx(90.0)

    def test_index_is_chronological(self, transformed_movies):
        result = aggregation.budget_and_box_office_by_decade(transformed_movies)
        assert list(result.index) == ['1990s', '2000s', '2010s', '2020s']

    def test_omits_decades_no_movie_occupies(self, tiny_transformed):
        """observed=True; without it, grouping a categorical yields empty rows."""
        result = aggregation.budget_and_box_office_by_decade(tiny_transformed)
        assert not result.isna().all(axis=1).any()

    def test_has_both_requested_columns(self, transformed_movies):
        result = aggregation.budget_and_box_office_by_decade(transformed_movies)
        assert list(result.columns) == ['Budget', 'BoxOffice']


class TestByCountry:

    def test_computes_mean_rating_and_totals(self, tiny_transformed):
        result = aggregation.statistics_by_country(tiny_transformed)
        # USA holds A, C, F: ratings 4.0, 7.1, 9.0; budgets 100, 200, 80.
        assert result.loc['USA', 'MeanRating'] == pytest.approx(6.7)
        assert result.loc['USA', 'TotalBudget'] == pytest.approx(380.0)
        assert result.loc['USA', 'TotalBoxOffice'] == pytest.approx(960.0)

    def test_is_sorted_by_box_office(self, transformed_movies):
        result = aggregation.statistics_by_country(transformed_movies)
        assert result['TotalBoxOffice'].is_monotonic_decreasing

    def test_covers_every_country(self, transformed_movies):
        result = aggregation.statistics_by_country(transformed_movies)
        assert set(result.index) == set(transformed_movies['Country'].unique())

    def test_totals_ignore_missing_values(self):
        frame = pd.DataFrame({
            'Country': ['X', 'X'], 'Rating': [8.0, np.nan],
            'Budget': [10.0, np.nan], 'BoxOffice': [20.0, 5.0],
        })
        result = aggregation.statistics_by_country(frame)
        assert result.loc['X', 'TotalBudget'] == pytest.approx(10.0)
        assert result.loc['X', 'MeanRating'] == pytest.approx(8.0)


class TestProfitablePercentageByGenre:

    def test_excludes_unknown_roi_from_the_denominator(self):
        """REGRESSION: the classroom version divided by len(group), charging
        every movie with a missing budget or box office against its genre as
        though it were unprofitable. Here Action has two known ROIs, both above
        the threshold, so the answer is 100% — not the 50% that counting the
        two NaN rows would give."""
        frame = pd.DataFrame({
            'Genre': ['Action'] * 4,
            'ROI': [2.0, 3.0, np.nan, np.nan],
        })
        result = aggregation.profitable_percentage_by_genre(frame)
        assert result['Action'] == pytest.approx(100.0)

    def test_reports_nan_when_no_roi_is_known(self):
        """'We cannot tell' must not be reported as 'none were profitable'."""
        frame = pd.DataFrame({'Genre': ['Horror'] * 3, 'ROI': [np.nan] * 3})
        result = aggregation.profitable_percentage_by_genre(frame)
        assert np.isnan(result['Horror'])

    def test_threshold_is_exclusive(self):
        frame = pd.DataFrame({'Genre': ['Drama'] * 2, 'ROI': [1.0, 1.1]})
        result = aggregation.profitable_percentage_by_genre(frame)
        assert result['Drama'] == pytest.approx(50.0)

    def test_threshold_is_configurable(self):
        frame = pd.DataFrame({'Genre': ['Drama'] * 4, 'ROI': [0.5, 1.5, 2.5, 3.5]})
        result = aggregation.profitable_percentage_by_genre(frame, threshold=2.0)
        assert result['Drama'] == pytest.approx(50.0)

    def test_computes_the_expected_share(self, tiny_transformed):
        # Action: A has ROI 2.0 (profitable), B has -0.2. One of two known.
        result = aggregation.profitable_percentage_by_genre(tiny_transformed)
        assert result['Action'] == pytest.approx(50.0)

    def test_stays_within_zero_and_one_hundred(self, transformed_movies):
        result = aggregation.profitable_percentage_by_genre(transformed_movies)
        assert result.dropna().between(0, 100).all()

    def test_is_sorted_descending(self, transformed_movies):
        result = aggregation.profitable_percentage_by_genre(transformed_movies)
        assert result.dropna().is_monotonic_decreasing

    def test_does_not_warn_about_grouping_columns(self, transformed_movies):
        """REGRESSION: groupby(...).apply(func) over the whole group frame emits
        a DeprecationWarning on pandas 2.2 and changes behaviour on pandas 3."""
        with warnings_as_errors():
            aggregation.profitable_percentage_by_genre(transformed_movies)


class TestAverageByYear:

    def test_is_indexed_by_year_in_order(self, transformed_movies):
        result = aggregation.average_by_year(transformed_movies)
        assert result.index.is_monotonic_increasing
        assert list(result.columns) == ['Budget', 'BoxOffice']

    def test_covers_every_year_present(self, transformed_movies):
        result = aggregation.average_by_year(transformed_movies)
        assert set(result.index) == set(transformed_movies['Year'].unique())


class TestNoDeprecationWarnings:
    """Every aggregation must be clean under pandas 2.2 and forward."""

    @pytest.mark.parametrize('name', [
        'average_rating_by_genre',
        'highest_grossing_by_director',
        'budget_and_box_office_by_decade',
        'statistics_by_country',
        'profitable_percentage_by_genre',
        'average_by_year',
    ])
    def test_aggregation_is_warning_free(self, transformed_movies, name):
        with warnings_as_errors():
            getattr(aggregation, name)(transformed_movies)
