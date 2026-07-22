"""Dataset generation — schema, reproducibility and missing-value injection."""

from __future__ import annotations

import numpy as np
import pytest

from pandas_homework import dataset


class TestGenerateDataset:

    def test_has_the_requested_number_of_rows(self):
        assert len(dataset.generate_dataset(n_movies=50)) == 50

    def test_has_the_expected_columns_in_order(self, raw_movies):
        assert tuple(raw_movies.columns) == dataset.RAW_COLUMNS

    def test_default_size_matches_the_brief(self, raw_movies):
        assert raw_movies.shape == (200, 9)

    def test_same_seed_reproduces_the_dataset(self):
        a = dataset.generate_dataset(n_movies=40, seed=3)
        b = dataset.generate_dataset(n_movies=40, seed=3)
        assert a.equals(b)

    def test_different_seeds_produce_different_datasets(self):
        a = dataset.generate_dataset(n_movies=40, seed=3)
        b = dataset.generate_dataset(n_movies=40, seed=4)
        assert not a.equals(b)

    def test_is_isolated_from_the_global_rng(self):
        """REGRESSION: the brief seeds the global RNG and then calls
        module-level np.random helpers, so anything else touching numpy's
        global state between seeding and generation changes the data."""
        a = dataset.generate_dataset(n_movies=40, seed=5)
        np.random.seed(999)
        np.random.random(100)
        b = dataset.generate_dataset(n_movies=40, seed=5)
        assert a.equals(b)

    def test_titles_are_unique_and_sequential(self):
        frame = dataset.generate_dataset(n_movies=10)
        assert list(frame['Title']) == [f'Movie {i}' for i in range(1, 11)]

    def test_years_fall_in_the_specified_range(self, raw_movies):
        assert raw_movies['Year'].between(1990, 2022).all()

    def test_categorical_columns_use_only_known_values(self, raw_movies):
        assert set(raw_movies['Genre']) <= set(dataset.GENRES)
        assert set(raw_movies['Country']) <= set(dataset.COUNTRIES)

    def test_directors_are_first_and_last_name_pairs(self, raw_movies):
        for name in raw_movies['Director']:
            first, last = name.split()
            assert first in dataset.DIRECTOR_FIRST_NAMES
            assert last in dataset.DIRECTOR_LAST_NAMES

    def test_box_office_derives_from_budget(self):
        """Box office is budget scaled by 0.5x-4x, so the ratio is bounded."""
        frame = dataset.generate_dataset(n_movies=200, missing_rate=0.0)
        ratio = frame['BoxOffice'] / frame['Budget']
        assert ratio.between(0.4, 4.1).all()


class TestMissingValues:

    def test_blanks_the_expected_number_per_column(self, raw_movies):
        for column in dataset.NULLABLE_COLUMNS:
            assert raw_movies[column].isna().sum() == 10

    def test_leaves_other_columns_complete(self, raw_movies):
        untouched = set(dataset.RAW_COLUMNS) - set(dataset.NULLABLE_COLUMNS)
        for column in untouched:
            assert raw_movies[column].notna().all()

    def test_zero_rate_produces_a_complete_frame(self):
        frame = dataset.generate_dataset(n_movies=50, missing_rate=0.0)
        assert frame.notna().all().all()

    def test_nullable_columns_are_float_dtype(self, raw_movies):
        """Ints cannot hold NaN; assigning into one raises under pandas 3."""
        for column in dataset.NULLABLE_COLUMNS:
            assert raw_movies[column].dtype == np.float64


class TestValidation:

    @pytest.mark.parametrize('n_movies', [0, -1])
    def test_rejects_non_positive_sizes(self, n_movies):
        with pytest.raises(ValueError, match='must be positive'):
            dataset.generate_dataset(n_movies=n_movies)

    @pytest.mark.parametrize('rate', [-0.1, 1.5])
    def test_rejects_out_of_range_missing_rates(self, rate):
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            dataset.generate_dataset(missing_rate=rate)


class TestPersistence:

    def test_round_trips_through_csv(self, tmp_path):
        frame = dataset.generate_dataset(n_movies=20, seed=1)
        path = dataset.save_dataset(frame, tmp_path / 'nested' / 'movies.csv')
        assert path.exists()
        assert dataset.load_dataset(path).equals(frame)

    def test_load_rejects_a_file_missing_required_columns(self, tmp_path):
        path = tmp_path / 'bad.csv'
        path.write_text('foo,bar\n1,2\n')
        with pytest.raises(ValueError, match='missing required column'):
            dataset.load_dataset(path)
