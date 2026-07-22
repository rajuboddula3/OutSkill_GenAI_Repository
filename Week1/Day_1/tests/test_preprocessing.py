"""Task 1 tests — preprocessing, stemming, lemmatisation."""

from __future__ import annotations

import pytest

from nlp_homework import preprocessing


class TestTokenize:

    def test_lowercases(self):
        assert preprocessing.tokenize('BRILLIANT Cinema') == ['brilliant', 'cinema']

    def test_removes_stopwords(self):
        assert 'the' not in preprocessing.tokenize('the film was the best')

    def test_removes_punctuation_tokens(self):
        assert preprocessing.tokenize('wow! amazing... film?') == ['wow', 'amazing', 'film']

    def test_removes_numbers(self):
        assert preprocessing.tokenize('rated 10 out of 10 stars') == ['rated', 'stars']

    def test_removes_alphanumeric_tokens_containing_digits(self):
        assert preprocessing.tokenize('the x11r5 release') == ['release']

    def test_collapses_extra_whitespace(self):
        assert preprocessing.tokenize('great    film\n\n  indeed') == ['great', 'film', 'indeed']

    def test_contraction_is_split_not_mangled(self):
        """REGRESSION: stripping punctuation before tokenising turned "don't"
        into "dont", which is not a stopword and survived as noise."""
        assert 'dont' not in preprocessing.tokenize("I don't like it")

    def test_dotted_token_is_split_not_concatenated(self):
        """REGRESSION: 'soc.religion.christian' became 'socreligionchristian'."""
        tokens = preprocessing.tokenize('posted to soc.religion.christian today')
        assert 'socreligionchristian' not in tokens

    def test_empty_string_yields_no_tokens(self):
        assert preprocessing.tokenize('') == []

    def test_stopwords_only_yields_no_tokens(self):
        assert preprocessing.tokenize('the and of a an is') == []


class TestLemmatization:

    @pytest.mark.parametrize('sentence,word,expected', [
        ('They were running fast', 'running', 'run'),
        ('The movies were long', 'movies', 'movie'),
        ('She studies film', 'studies', 'study'),
    ])
    def test_pos_aware_lemmatization_resolves_inflections(self, sentence, word, expected):
        """REGRESSION: untagged WordNetLemmatizer defaults to noun, leaving
        'running' as 'running'. The notebook's prose claimed otherwise."""
        tokens = sentence.lower().split()
        lemmas = dict(zip(tokens, preprocessing.lemmatize_tokens(tokens)))
        assert lemmas[word] == expected

    def test_empty_input_returns_empty_list(self):
        assert preprocessing.lemmatize_tokens([]) == []

    def test_output_length_matches_input_length(self):
        tokens = ['the', 'films', 'were', 'running', 'well']
        assert len(preprocessing.lemmatize_tokens(tokens)) == len(tokens)


class TestStemming:

    @pytest.mark.parametrize('word,expected', [
        ('running', 'run'), ('studies', 'studi'), ('movies', 'movi'),
    ])
    def test_porter_stems(self, word, expected):
        assert preprocessing.stem_tokens([word]) == [expected]


class TestPreprocess:

    def test_returns_all_products_with_consistent_lengths(self):
        result = preprocessing.preprocess('The films were absolutely brilliant!')
        assert len(result.tokens) == len(result.stemmed) == len(result.lemmatized)

    def test_cleaned_text_is_the_tokens_rejoined(self):
        result = preprocessing.preprocess('The films were absolutely brilliant!')
        assert result.cleaned == ' '.join(result.tokens)

    def test_cleaned_text_has_no_double_spaces(self):
        result = preprocessing.preprocess('great    film    indeed')
        assert '  ' not in result.cleaned

    def test_word_count_matches_token_count(self):
        result = preprocessing.preprocess('The films were absolutely brilliant!')
        assert result.word_count == len(result.tokens)

    def test_handles_empty_document(self):
        result = preprocessing.preprocess('')
        assert result.tokens == [] and result.cleaned == '' and result.word_count == 0


class TestComparisonNarrative:
    """The discussion text must be derived from results, not hardcoded."""

    def test_comparison_reports_all_three_forms(self):
        rows = preprocessing.compare_normalizers(['running', 'studies'])
        assert [r['original'] for r in rows] == ['running', 'studies']
        assert all({'original', 'stemmed', 'lemmatized'} == set(r) for r in rows)

    def test_narrative_quotes_the_actual_computed_values(self):
        """REGRESSION: the original printed a fixed paragraph that contradicted
        its own table (it claimed 'running' lemmatises to 'run' untagged)."""
        rows = preprocessing.compare_normalizers(['studies'])
        notes = ' '.join(preprocessing.describe_comparison(rows))
        assert rows[0]['stemmed'] in notes
        assert rows[0]['lemmatized'] in notes

    def test_narrative_adapts_to_different_inputs(self):
        a = preprocessing.describe_comparison(preprocessing.compare_normalizers(['studies']))
        b = preprocessing.describe_comparison(preprocessing.compare_normalizers(['movies']))
        assert a != b

    def test_narrative_handles_a_word_both_normalisers_leave_alone(self):
        notes = preprocessing.describe_comparison(
            preprocessing.compare_normalizers(['film']))
        assert notes  # must not raise or return nothing


class TestStopwordCaching:

    def test_stopword_set_is_loaded_once(self):
        """REGRESSION: the original rebuilt this set inside the per-document
        function, re-reading the corpus for all 1000 reviews."""
        preprocessing._stopword_set.cache_clear()
        preprocessing._stopword_set()
        preprocessing._stopword_set()
        assert preprocessing._stopword_set.cache_info().misses == 1
