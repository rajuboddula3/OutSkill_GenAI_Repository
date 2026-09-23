"""End-to-end tests — every task runs against one corpus and produces artefacts.

These use a 20-review corpus built from in-repo filler text, so the suite needs
no network access and finishes in seconds.
"""

from __future__ import annotations

import re

import pytest

from nlp_homework import pipeline


@pytest.fixture(scope='module')
def task_one(small_corpus):
    return pipeline.run_task_one(small_corpus)


@pytest.fixture(scope='module')
def task_two(task_one, tmp_path_factory):
    return pipeline.run_task_two(task_one.frame, tmp_path_factory.mktemp('task2'))


@pytest.fixture(scope='module')
def task_three(task_one, tmp_path_factory):
    return pipeline.run_task_three(task_one.frame, tmp_path_factory.mktemp('task3'),
                                   sample_size=20)


class TestTaskOne:

    def test_adds_every_derived_column(self, task_one):
        expected = {'tokens', 'preprocessed', 'stemmed', 'lemmatized', 'word_count'}
        assert expected <= set(task_one.frame.columns)

    def test_preserves_the_original_columns_and_row_count(self, task_one, small_corpus):
        assert len(task_one.frame) == len(small_corpus)
        assert task_one.frame['review'].equals(small_corpus['review'])

    def test_token_columns_are_length_consistent_per_row(self, task_one):
        for _, row in task_one.frame.iterrows():
            assert len(row['tokens']) == len(row['stemmed']) == len(row['lemmatized'])

    def test_produces_a_discussion_grounded_in_the_comparison(self, task_one):
        assert task_one.comparison and task_one.discussion


class TestTaskTwo:
    """Covers the five Task 2 deliverables from the brief."""

    def test_reports_average_length_and_vocabulary_size(self, task_two):
        assert task_two.statistics['average_length'] > 0
        assert task_two.statistics['vocabulary_size'] > 0

    def test_separates_common_words_by_sentiment(self, task_two):
        assert task_two.positive_words and task_two.negative_words
        assert task_two.positive_words != task_two.negative_words

    def test_produces_both_bigrams_and_trigrams(self, task_two):
        assert task_two.positive_bigrams and task_two.positive_trigrams
        assert all(len(g.split()) == 2 for g, _ in task_two.positive_bigrams)
        assert all(len(g.split()) == 3 for g, _ in task_two.positive_trigrams)

    def test_produces_tfidf_terms_for_both_classes(self, task_two):
        assert task_two.positive_tfidf and task_two.negative_tfidf

    def test_writes_all_six_figures(self, task_two):
        assert len(task_two.figures) == 6
        assert all(f.exists() and f.stat().st_size > 0 for f in task_two.figures)


class TestTaskThree:
    """Covers the seven Task 3 deliverables from the brief."""

    def test_extracts_entities_with_offsets(self, task_three):
        assert not task_three.entities.empty
        assert {'entity_type', 'entity_text', 'start', 'end'} <= set(task_three.entities.columns)

    def test_entity_offsets_point_at_the_entity_text(self, task_three, task_one):
        for _, row in task_three.entities.head(30).iterrows():
            source = task_one.frame.loc[row['review_id'], 'review']
            assert source[row['start']:row['end']] == row['entity_text']

    def test_records_sentiment_alongside_each_entity(self, task_three):
        assert set(task_three.entities['sentiment'].unique()) <= {0, 1}

    def test_finds_the_injected_domain_entities(self, task_three):
        assert not task_three.custom_entities.empty
        assert set(task_three.custom_entities['entity_type']) <= {
            'DIRECTOR', 'ACTOR', 'MOVIE', 'AWARD', 'POTENTIAL_MOVIE'}

    def test_writes_the_four_ner_figures(self, task_three):
        assert len(task_three.figures) == 4
        assert all(f.exists() and f.stat().st_size > 0 for f in task_three.figures)

    def test_reports_overall_per_label_and_per_category_metrics(self, task_three):
        assert set(task_three.overall_metrics) >= {'precision', 'recall', 'f1'}
        assert task_three.per_label_metrics
        assert {'negative', 'unseen_entity'} <= set(task_three.category_metrics)

    def test_produces_highlighted_html_for_sample_reviews(self, task_three):
        """Task 3 item 7 — the original computed entities but printed a
        placeholder string instead of rendering the highlighted text."""
        assert len(task_three.highlighted_samples) == 3
        for _, html_out in task_three.highlighted_samples:
            assert '<span style="background-color:' in html_out

    def test_highlighting_preserves_the_source_text(self, task_three, task_one):
        import html as html_module
        for review_id, html_out in task_three.highlighted_samples:
            stripped = html_module.unescape(re.sub(r'<[^>]+>', '', html_out))
            assert stripped == task_one.frame.loc[review_id, 'review']


class TestFullRun:

    def test_runs_end_to_end_and_writes_the_dataset(self, tmp_path, monkeypatch, base_texts):
        """The whole assignment, from generation to figures, in one call."""
        monkeypatch.setattr(pipeline, 'generate_dataset',
                            lambda n_reviews, seed: __import__(
                                'nlp_homework.dataset', fromlist=['generate_dataset']
                            ).generate_dataset(n_reviews=n_reviews, seed=seed,
                                               base_texts=base_texts))
        csv_path = tmp_path / 'movie_reviews.csv'
        result = pipeline.run(output_dir=tmp_path, n_reviews=20, sample_size=10,
                              dataset_path=csv_path, download_resources=False)

        assert csv_path.exists()
        assert len(result.dataset) == 20
        assert len(result.task_two.figures) == 6
        assert result.task_three.highlighted_samples
        assert all(f.exists() for f in result.task_two.figures + result.task_three.figures)
