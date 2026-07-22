"""Dataset generation tests — shape, balance and reproducibility."""

from __future__ import annotations

import random

import pytest

from nlp_homework import dataset
from nlp_homework.vocabulary import NEGATIVE_WORDS, POSITIVE_WORDS


class TestGenerateDataset:

    def test_has_the_requested_number_of_rows(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=20, seed=1, base_texts=base_texts)
        assert len(frame) == 20

    def test_is_balanced(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=20, seed=1, base_texts=base_texts)
        assert frame['sentiment'].value_counts().to_dict() == {0: 10, 1: 10}

    def test_has_the_expected_columns(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=10, seed=1, base_texts=base_texts)
        assert list(frame.columns) == ['review', 'sentiment']

    def test_same_seed_reproduces_the_corpus(self, base_texts):
        """REGRESSION: the original shuffled with `sample(frac=1)` and no
        random_state, so the corpus was not reproducible."""
        a = dataset.generate_dataset(n_reviews=20, seed=3, base_texts=base_texts)
        b = dataset.generate_dataset(n_reviews=20, seed=3, base_texts=base_texts)
        assert a.equals(b)

    def test_different_seeds_produce_different_corpora(self, base_texts):
        a = dataset.generate_dataset(n_reviews=20, seed=3, base_texts=base_texts)
        b = dataset.generate_dataset(n_reviews=20, seed=4, base_texts=base_texts)
        assert not a.equals(b)

    def test_is_isolated_from_the_global_rng(self, base_texts):
        """Generation must not depend on module-level random state."""
        random.seed(999)
        a = dataset.generate_dataset(n_reviews=20, seed=5, base_texts=base_texts)
        random.seed(111)
        [random.random() for _ in range(50)]
        b = dataset.generate_dataset(n_reviews=20, seed=5, base_texts=base_texts)
        assert a.equals(b)

    def test_rejects_odd_review_counts(self, base_texts):
        with pytest.raises(ValueError, match='even'):
            dataset.generate_dataset(n_reviews=7, base_texts=base_texts)

    def test_rejects_empty_base_texts(self):
        with pytest.raises(ValueError, match='empty'):
            dataset.generate_dataset(n_reviews=4, base_texts=[])

    def test_reviews_are_non_empty(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=20, seed=1, base_texts=base_texts)
        assert frame['review'].str.strip().str.len().gt(0).all()

    def test_reviews_contain_no_newlines_or_double_spaces(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=20, seed=1, base_texts=base_texts)
        assert not frame['review'].str.contains(r'\n|  ', regex=True).any()

    def test_sentiment_words_are_injected_by_polarity(self, base_texts):
        frame = dataset.generate_dataset(n_reviews=40, seed=2, base_texts=base_texts)
        for _, row in frame.iterrows():
            pool = POSITIVE_WORDS if row['sentiment'] == 1 else NEGATIVE_WORDS
            words = set(row['review'].lower().split())
            assert words & set(pool), f'no polarity words in: {row["review"][:80]}'


class TestGenerateReview:

    def test_respects_the_length_ceiling_before_injection(self, base_texts):
        rng = random.Random(0)
        review = dataset.generate_review(1, base_texts, rng, length_range=(5, 5))
        # 5 filler words + at most 10 sentiment + 5 movie terms + ~9 entities
        assert 5 <= len(review.split()) <= 60

    def test_handles_a_single_word_base_text(self):
        rng = random.Random(0)
        assert dataset.generate_review(1, ['solo'], rng, length_range=(1, 1)).strip()


class TestPersistence:

    def test_round_trips_through_csv(self, tmp_path, base_texts):
        frame = dataset.generate_dataset(n_reviews=10, seed=1, base_texts=base_texts)
        path = dataset.save_dataset(frame, tmp_path / 'nested' / 'reviews.csv')
        assert path.exists()
        reloaded = dataset.load_dataset(path)
        assert reloaded.equals(frame)

    def test_load_rejects_a_file_missing_required_columns(self, tmp_path):
        path = tmp_path / 'bad.csv'
        path.write_text('foo,bar\n1,2\n')
        with pytest.raises(ValueError, match='missing required column'):
            dataset.load_dataset(path)
