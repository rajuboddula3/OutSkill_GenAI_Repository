"""Task 3 — derived columns, band boundaries and categorical dtypes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pandas_homework import transformation


class TestProfit:

    def test_is_box_office_minus_budget(self, tiny_movies):
        profit = transformation.add_profit(tiny_movies)['Profit']
        assert profit.iloc[0] == pytest.approx(200.0)  # 300 - 100
        assert profit.iloc[1] == pytest.approx(-10.0)  # 40 - 50

    def test_is_nan_when_either_input_is_missing(self, tiny_movies):
        profit = transformation.add_profit(tiny_movies)['Profit']
        assert np.isnan(profit.iloc[3])  # D has no budget
        assert np.isnan(profit.iloc[4])  # E has no box office


class TestROI:

    def test_uses_the_formula_from_the_brief(self, tiny_movies):
        roi = transformation.add_roi(tiny_movies)['ROI']
        assert roi.iloc[0] == pytest.approx(2.0)   # (300-100)/100
        assert roi.iloc[1] == pytest.approx(-0.2)  # (40-50)/50

    def test_is_nan_when_either_input_is_missing(self, tiny_movies):
        roi = transformation.add_roi(tiny_movies)['ROI']
        assert np.isnan(roi.iloc[3])
        assert np.isnan(roi.iloc[4])

    def test_zero_budget_yields_nan_not_infinity(self, tiny_movies):
        """An infinite ROI would silently poison every downstream mean."""
        frame = tiny_movies.assign(Budget=tiny_movies['Budget'].mask(
            tiny_movies['Title'] == 'A', 0.0))
        roi = transformation.add_roi(frame)['ROI']
        assert np.isnan(roi.iloc[0])
        assert np.isfinite(roi.dropna()).all()


class TestLengthBand:

    @pytest.mark.parametrize('runtime, expected', [
        (74, 'Short'),
        (89, 'Short'),
        (89.9, 'Short'),
        (90, 'Medium'),    # lower edge of the brief's 90-120 band
        (105, 'Medium'),
        (120, 'Medium'),   # upper edge is inclusive
        (120.5, 'Long'),
        (121, 'Long'),
        (200, 'Long'),
    ])
    def test_band_boundaries(self, tiny_movies, runtime, expected):
        frame = tiny_movies.head(1).assign(Runtime=[float(runtime)])
        assert transformation.add_length_band(frame)['Length'].iloc[0] == expected

    def test_missing_runtime_stays_missing(self, tiny_movies):
        length = transformation.add_length_band(tiny_movies)['Length']
        assert pd.isna(length.iloc[4])  # E has no runtime

    def test_is_an_ordered_categorical(self, tiny_movies):
        length = transformation.add_length_band(tiny_movies)['Length']
        assert isinstance(length.dtype, pd.CategoricalDtype)
        assert length.cat.ordered
        assert list(length.cat.categories) == ['Short', 'Medium', 'Long']

    def test_ordering_is_by_duration_not_alphabet(self, tiny_movies):
        length = transformation.add_length_band(tiny_movies)['Length']
        assert length.min() == 'Short'
        assert length.max() == 'Long'


class TestDecade:

    def test_labels_the_decade(self, tiny_movies):
        decade = transformation.add_decade(tiny_movies)['Decade']
        assert list(decade) == ['1990s', '2000s', '2010s', '2010s', '2020s', '1990s']

    def test_is_an_ordered_categorical(self, tiny_movies):
        decade = transformation.add_decade(tiny_movies)['Decade']
        assert isinstance(decade.dtype, pd.CategoricalDtype)
        assert decade.cat.ordered

    def test_categories_are_chronological(self, tiny_movies):
        decade = transformation.add_decade(tiny_movies)['Decade']
        assert list(decade.cat.categories) == ['1990s', '2000s', '2010s', '2020s']

    def test_categories_cover_only_observed_decades(self, tiny_movies):
        decade = transformation.add_decade(tiny_movies)['Decade']
        assert '1980s' not in decade.cat.categories

    def test_ordering_survives_three_digit_years(self):
        """REGRESSION: the classroom version built plain strings and relied on
        lexicographic sorting, which puts '980s' after '1990s'."""
        frame = pd.DataFrame({'Year': [1990, 980, 2010]})
        decade = transformation.add_decade(frame)['Decade']
        assert list(decade.cat.categories) == ['980s', '1990s', '2010s']
        assert decade.min() == '980s'


class TestRatingBand:

    @pytest.mark.parametrize('rating, expected', [
        (0.0, 'Poor'),
        (3.9, 'Poor'),
        (4.0, 'Poor'),       # upper edge of Poor is inclusive
        (4.1, 'Average'),
        (7.0, 'Average'),    # upper edge of Average is inclusive
        (7.1, 'Excellent'),
        (10.0, 'Excellent'),
    ])
    def test_band_boundaries(self, tiny_movies, rating, expected):
        frame = tiny_movies.head(1).assign(Rating=[rating])
        assert transformation.add_rating_band(frame)['RatingBand'].iloc[0] == expected

    def test_missing_rating_stays_missing(self, tiny_movies):
        band = transformation.add_rating_band(tiny_movies)['RatingBand']
        assert pd.isna(band.iloc[3])

    def test_is_an_ordered_categorical(self, tiny_movies):
        """REGRESSION: the brief says 'convert to a categorical type'. The
        classroom version produced an object column of plain strings."""
        band = transformation.add_rating_band(tiny_movies)['RatingBand']
        assert isinstance(band.dtype, pd.CategoricalDtype)
        assert band.cat.ordered
        assert list(band.cat.categories) == ['Poor', 'Average', 'Excellent']

    def test_ordering_is_by_quality_not_alphabet(self, tiny_movies):
        band = transformation.add_rating_band(tiny_movies)['RatingBand']
        assert band.min() == 'Poor'
        assert band.max() == 'Excellent'

    def test_keeps_the_numeric_rating_column(self, tiny_movies):
        """Tasks 4 and 5 still need the numeric ratings."""
        result = transformation.add_rating_band(tiny_movies)
        assert result['Rating'].dtype == np.float64
        assert result['Rating'].iloc[0] == 4.0


class TestTransform:

    def test_adds_every_derived_column(self, tiny_transformed):
        for column in transformation.DERIVED_COLUMNS:
            assert column in tiny_transformed.columns

    def test_keeps_every_original_column(self, tiny_movies, tiny_transformed):
        assert set(tiny_movies.columns) <= set(tiny_transformed.columns)

    def test_preserves_row_count_and_index(self, tiny_movies, tiny_transformed):
        assert len(tiny_transformed) == len(tiny_movies)
        assert tiny_transformed.index.equals(tiny_movies.index)

    def test_does_not_mutate_the_input(self, tiny_movies):
        """REGRESSION: the classroom notebook added columns to movies_data in
        place across five cells, so re-running one changed later results."""
        before = tiny_movies.copy()
        transformation.transform(tiny_movies)
        assert tiny_movies.equals(before)
        assert 'Profit' not in tiny_movies.columns

    def test_is_idempotent(self, tiny_movies):
        once = transformation.transform(tiny_movies)
        twice = transformation.transform(once)
        assert once.equals(twice)

    def test_works_on_the_full_generated_dataset(self, transformed_movies):
        assert len(transformed_movies) == 200
        assert transformed_movies['ROI'].notna().sum() > 150
