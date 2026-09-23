"""Task 3 tests.

Several of these are regression tests for defects in the original notebook
implementation; those are marked with a REGRESSION comment naming the defect.
"""

from __future__ import annotations

import pytest

from nlp_homework import ner
from nlp_homework.evaluation_data import TEST_REVIEWS, subset


# ---------------------------------------------------------------------------
# custom_movie_ner
# ---------------------------------------------------------------------------

class TestCustomMovieNER:

    def test_finds_every_mention_not_just_the_first(self):
        """REGRESSION: the original used a single str.find() per dictionary term,
        so only the first mention of 'Tom Hanks' was ever reported."""
        text = 'Tom Hanks was great. Later Tom Hanks appeared again with Tom Hanks.'
        actors = [e for e in ner.custom_movie_ner(text) if e.label == 'ACTOR']
        assert len(actors) == 3
        assert {e.start for e in actors} == {0, 27, 57}

    @pytest.mark.parametrize('text', [
        'She collects Oscars memorabilia.',      # 'Oscar' inside 'Oscars'
        'He admired Avatar-like visual design.',  # 'Avatar' inside a compound
        'The Titanics sank in the sequel.',       # 'Titanic' inside 'Titanics'
    ])
    def test_rejects_substring_matches(self, text):
        """REGRESSION: naive `term.lower() in text.lower()` had no word boundary."""
        assert ner.custom_movie_ner(text) == []

    @pytest.mark.parametrize('text,label,expected', [
        ('Steven Spielberg directed it.', 'DIRECTOR', 'Steven Spielberg'),
        ('Starring Samuel L. Jackson.', 'ACTOR', 'Samuel L. Jackson'),
        ('I rewatched Pulp Fiction.', 'MOVIE', 'Pulp Fiction'),
        ('It won an Oscar.', 'AWARD', 'Oscar'),
    ])
    def test_recognises_each_gazetteer_category(self, text, label, expected):
        matches = [e for e in ner.custom_movie_ner(text) if e.label == label]
        assert [e.text for e in matches] == [expected]

    def test_hyphenated_name_survives_boundary_check(self):
        """The boundary class contains '-', so verify an internal hyphen still matches."""
        entities = ner.custom_movie_ner('Daniel Day-Lewis retired from acting.')
        assert [(e.label, e.text) for e in entities] == [('ACTOR', 'Daniel Day-Lewis')]

    def test_possessive_does_not_block_match(self):
        entities = ner.custom_movie_ner("Quentin Tarantino's next film.")
        assert ('DIRECTOR', 'Quentin Tarantino') in [(e.label, e.text) for e in entities]

    def test_prefers_longest_award_name(self):
        """'Academy Award' must win over any shorter overlapping award term."""
        entities = ner.custom_movie_ner('He won an Academy Award last night.')
        assert [(e.label, e.text) for e in entities] == [('AWARD', 'Academy Award')]

    def test_matching_is_case_insensitive_but_preserves_source_casing(self):
        entities = ner.custom_movie_ner('i loved pulp fiction honestly')
        assert [(e.label, e.text) for e in entities] == [('MOVIE', 'pulp fiction')]

    def test_quoted_capitalised_phrase_becomes_potential_movie(self):
        entities = ner.custom_movie_ner('We watched "The Silent Hour" last night.')
        assert ('POTENTIAL_MOVIE', 'The Silent Hour') in [(e.label, e.text) for e in entities]

    def test_quoted_phrase_already_in_gazetteer_is_not_duplicated(self):
        labels = [e.label for e in ner.custom_movie_ner('We watched "Inception" again.')]
        assert labels.count('POTENTIAL_MOVIE') == 0
        assert 'MOVIE' in labels

    def test_returns_spans_that_index_back_into_the_source(self):
        text = 'Avatar and Titanic were both directed by James Cameron.'
        for entity in ner.custom_movie_ner(text):
            assert text[entity.start:entity.end] == entity.text

    def test_empty_text_yields_no_entities(self):
        assert ner.custom_movie_ner('') == []


# ---------------------------------------------------------------------------
# Overlap resolution and highlighting
# ---------------------------------------------------------------------------

class TestOverlapResolution:

    def test_domain_label_beats_generic_label_on_identical_span(self):
        """REGRESSION: NLTK tags 'Steven Spielberg' PERSON while the custom NER
        tags it DIRECTOR. The original emitted both, duplicating the text."""
        entities = [
            ner.Entity(0, 16, 'PERSON', 'Steven Spielberg'),
            ner.Entity(0, 16, 'DIRECTOR', 'Steven Spielberg'),
        ]
        assert [e.label for e in ner.resolve_overlaps(entities)] == ['DIRECTOR']

    def test_longer_span_wins(self):
        entities = [
            ner.Entity(0, 5, 'AWARD', 'Award'),
            ner.Entity(0, 13, 'AWARD', 'Academy Award'),
        ]
        assert [e.text for e in ner.resolve_overlaps(entities)] == ['Academy Award']

    def test_disjoint_entities_all_survive(self):
        entities = [
            ner.Entity(0, 6, 'MOVIE', 'Avatar'),
            ner.Entity(11, 18, 'MOVIE', 'Titanic'),
        ]
        assert len(ner.resolve_overlaps(entities)) == 2

    def test_adjacent_touching_spans_both_survive(self):
        """end == start is adjacency, not overlap."""
        entities = [ner.Entity(0, 5, 'A', 'aaaaa'), ner.Entity(5, 10, 'B', 'bbbbb')]
        assert len(ner.resolve_overlaps(entities)) == 2


class TestHighlighting:

    def test_source_text_is_preserved_exactly_when_stripped_of_markup(self):
        """REGRESSION: overlapping spans used to duplicate substrings."""
        import re
        text = "Steven Spielberg's Jurassic Park won an Oscar."
        html_out = ner.highlight_entities(text, ner.extract_all_entities(text))
        assert re.sub(r'<[^>]+>', '', html_out) == text.replace("'", '&#x27;')

    def test_no_entity_appears_twice(self):
        text = "Steven Spielberg's Jurassic Park won an Oscar."
        html_out = ner.highlight_entities(text, ner.extract_all_entities(text))
        assert html_out.count('Steven Spielberg') == 1

    def test_escapes_html_metacharacters_in_surrounding_text(self):
        """REGRESSION: raw text was interpolated into markup unescaped."""
        html_out = ner.highlight_entities('Avatar & <b>bold</b>', [ner.Entity(0, 6, 'MOVIE', 'Avatar')])
        assert '<b>' not in html_out
        assert '&amp;' in html_out and '&lt;b&gt;' in html_out

    def test_escapes_metacharacters_inside_an_entity_span(self):
        html_out = ner.highlight_entities('<script>', [ner.Entity(0, 8, 'MOVIE', '<script>')])
        assert '<script>' not in html_out
        assert '&lt;script&gt;' in html_out

    def test_applies_the_configured_colour_per_label(self):
        html_out = ner.highlight_entities('Avatar', [ner.Entity(0, 6, 'MOVIE', 'Avatar')])
        assert ner.ENTITY_COLORS['MOVIE'] in html_out

    def test_unknown_label_falls_back_to_default_colour(self):
        html_out = ner.highlight_entities('Avatar', [ner.Entity(0, 6, 'WHATEVER', 'Avatar')])
        assert ner._DEFAULT_COLOR in html_out

    def test_text_without_entities_is_returned_escaped_and_unwrapped(self):
        assert ner.highlight_entities('plain & simple', []) == 'plain &amp; simple'


# ---------------------------------------------------------------------------
# NLTK extractor
# ---------------------------------------------------------------------------

class TestNLTKExtractor:

    def test_offsets_index_back_into_the_source(self):
        text = 'Barack Obama visited Paris and met Angela Merkel in Paris again.'
        for entity in ner.extract_entities_nltk(text):
            assert text[entity.start:entity.end] == entity.text

    def test_repeated_mentions_get_distinct_offsets(self):
        text = 'Paris is lovely. Paris is crowded. Paris is expensive.'
        starts = [e.start for e in ner.extract_entities_nltk(text)]
        assert len(starts) == len(set(starts))

    def test_returns_empty_list_for_text_without_entities(self):
        assert ner.extract_entities_nltk('the cat sat on the mat') == []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_perfect_prediction_scores_one(self):
        data = [{'text': 'Avatar', 'expected': [('MOVIE', 'Avatar')]}]
        metrics = ner.evaluate(data, lambda t: [ner.Entity(0, 6, 'MOVIE', 'Avatar')])
        assert metrics['precision'] == metrics['recall'] == metrics['f1'] == 1.0

    def test_prediction_on_a_negative_example_is_a_false_positive(self):
        """The original test set had no negatives, so precision could not drop."""
        data = [{'text': 'nothing here', 'expected': []}]
        metrics = ner.evaluate(data, lambda t: [ner.Entity(0, 7, 'MOVIE', 'nothing')])
        assert metrics['false_positives'] == 1
        assert metrics['precision'] == 0.0

    def test_empty_prediction_and_empty_truth_scores_zero_without_dividing_by_zero(self):
        metrics = ner.evaluate([{'text': 'x', 'expected': []}], lambda t: [])
        assert metrics == {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                           'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}

    def test_missed_entity_is_a_false_negative(self):
        data = [{'text': 'Avatar', 'expected': [('MOVIE', 'Avatar')]}]
        metrics = ner.evaluate(data, lambda t: [])
        assert metrics['false_negatives'] == 1
        assert metrics['recall'] == 0.0

    def test_per_label_metrics_cover_every_label_in_the_truth_set(self):
        by_label = ner.evaluate_by_label(TEST_REVIEWS, ner.custom_movie_ner)
        assert {'DIRECTOR', 'ACTOR', 'MOVIE', 'AWARD'} <= set(by_label)


class TestHonestEvaluationSet:
    """The evaluation set must actually be able to expose weaknesses."""

    def test_contains_negative_and_unseen_entity_cases(self):
        categories = {e['category'] for e in TEST_REVIEWS}
        assert {'negative', 'unseen_entity', 'substring_trap'} <= categories

    def test_gazetteer_ner_scores_perfectly_on_in_gazetteer_cases(self):
        metrics = ner.evaluate(subset('in_gazetteer'), ner.custom_movie_ner)
        assert metrics['f1'] == 1.0

    def test_gazetteer_ner_cannot_recall_unseen_entities(self):
        """A dictionary recogniser must miss entities outside its dictionary.
        If this ever passes, the gazetteer has silently absorbed the test set."""
        metrics = ner.evaluate(subset('unseen_entity'), ner.custom_movie_ner)
        assert metrics['recall'] < 1.0

    def test_no_false_positives_on_negatives_or_traps(self):
        for category in ('negative', 'substring_trap'):
            metrics = ner.evaluate(subset(category), ner.custom_movie_ner)
            assert metrics['false_positives'] == 0, category

    def test_overall_f1_is_not_a_perfect_score(self):
        """REGRESSION: the original scored F1 = 1.00 because it was evaluated
        against the same gazetteer that produced the predictions."""
        assert ner.evaluate(TEST_REVIEWS, ner.custom_movie_ner)['f1'] < 1.0
