"""Task 2 tests — statistics, n-grams, TF-IDF and figure output."""

from __future__ import annotations

import pandas as pd
import pytest

from nlp_homework import exploration


@pytest.fixture
def token_lists():
    return pd.Series([
        ['great', 'film', 'great'],
        ['awful', 'film'],
        ['great', 'movie', 'plot', 'twist'],
    ])


class TestCorpusStatistics:

    def test_reports_document_count_and_lengths(self, token_lists):
        stats = exploration.corpus_statistics(token_lists)
        assert stats['n_documents'] == 3
        assert stats['average_length'] == pytest.approx(3.0)
        assert stats['min_length'] == 2 and stats['max_length'] == 4

    def test_vocabulary_counts_unique_tokens_only(self, token_lists):
        # great, film, awful, movie, plot, twist
        assert exploration.corpus_statistics(token_lists)['vocabulary_size'] == 6

    def test_total_tokens_counts_every_occurrence(self, token_lists):
        assert exploration.corpus_statistics(token_lists)['total_tokens'] == 9


class TestMostCommonWords:

    def test_ranks_by_frequency(self, token_lists):
        assert exploration.most_common_words(token_lists, top_n=1) == [('great', 3)]

    def test_respects_top_n(self, token_lists):
        assert len(exploration.most_common_words(token_lists, top_n=2)) == 2


class TestNgramFrequencies:

    def test_builds_bigrams_as_joined_strings(self):
        series = pd.Series([['great', 'film'], ['great', 'film']])
        assert exploration.ngram_frequencies(series, 2) == [('great film', 2)]

    def test_ngrams_never_span_a_document_boundary(self):
        """'film awful' would only exist if two documents were concatenated."""
        series = pd.Series([['great', 'film'], ['awful', 'plot']])
        assert 'film awful' not in dict(exploration.ngram_frequencies(series, 2))

    def test_document_shorter_than_n_contributes_nothing(self):
        assert exploration.ngram_frequencies(pd.Series([['solo']]), 3) == []


class TestTfidf:

    @pytest.fixture
    def corpus(self):
        documents = pd.Series(['great film wonderful', 'awful film terrible',
                               'great movie wonderful', 'awful plot terrible'])
        return documents, pd.Series([1, 0, 1, 0])

    def test_surfaces_class_specific_terms(self, corpus):
        documents, labels = corpus
        top = dict(exploration.top_tfidf_terms(documents, labels, 1, top_n=3))
        assert 'great' in top and 'wonderful' in top

    def test_scores_are_sorted_descending(self, corpus):
        documents, labels = corpus
        scores = [s for _, s in exploration.top_tfidf_terms(documents, labels, 1, top_n=5)]
        assert scores == sorted(scores, reverse=True)

    def test_absent_label_returns_empty(self, corpus):
        documents, labels = corpus
        assert exploration.top_tfidf_terms(documents, labels, 99) == []


class TestFigures:

    def test_length_distribution_writes_a_png(self, tmp_path, token_lists):
        path = exploration.plot_length_distribution(token_lists, tmp_path / 'len.png')
        assert path.exists() and path.stat().st_size > 0

    def test_creates_missing_parent_directories(self, tmp_path, token_lists):
        path = exploration.plot_length_distribution(token_lists, tmp_path / 'a' / 'b' / 'len.png')
        assert path.exists()

    def test_wordcloud_writes_a_png(self, tmp_path):
        path = exploration.plot_wordcloud(['great'] * 20, 'T', tmp_path / 'wc.png')
        assert path is not None and path.exists()

    def test_wordcloud_returns_none_for_empty_input(self, tmp_path):
        assert exploration.plot_wordcloud([], 'T', tmp_path / 'wc.png') is None

    def test_ranked_bars_writes_a_png(self, tmp_path):
        path = exploration.plot_ranked_bars(
            [('A', [('x', 3), ('y', 1)]), ('B', [('z', 2)])], tmp_path / 'bars.png')
        assert path.exists()

    def test_figures_are_closed_so_handles_do_not_leak(self, tmp_path, token_lists):
        """REGRESSION: the notebook never closed figures, so a long run
        accumulated open handles and matplotlib emitted a memory warning."""
        import matplotlib.pyplot as plt
        plt.close('all')
        for i in range(12):
            exploration.plot_length_distribution(token_lists, tmp_path / f'{i}.png')
        assert plt.get_fignums() == []
