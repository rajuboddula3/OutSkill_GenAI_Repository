"""Manually labelled test set for the custom NER (Task 3, item 6).

The brief's own sample test set contained only positive examples drawn from the
same gazetteer that powers the recogniser, so it scored a meaningless F1 = 1.00.
This set deliberately includes cases the dictionary *cannot* get right:

* ``unseen_entity`` — real entities absent from the gazetteer. A dictionary
  recogniser must miss these; they establish the honest recall ceiling.
* ``negative`` — no entities at all. Any prediction is a false positive.
* ``substring_trap`` — near-misses ('Oscars', 'Avatar-like') that a naive
  substring search would wrongly fire on.

Note: :func:`ner.evaluate` compares sets of ``(label, text)`` pairs, per the
brief. That collapses repeated mentions of the same entity within one document,
so this set cannot measure repeat-mention recall — ``tests/test_ner.py`` covers
that directly against :func:`ner.custom_movie_ner` instead.
"""

from __future__ import annotations

TEST_REVIEWS: list[dict] = [
    {
        'text': "Steven Spielberg directed 'Jurassic Park' which won an Oscar for special effects.",
        'expected': [('DIRECTOR', 'Steven Spielberg'), ('MOVIE', 'Jurassic Park'), ('AWARD', 'Oscar')],
        'category': 'in_gazetteer',
    },
    {
        'text': 'I thought The Dark Knight was brilliant with amazing performances by Christian Bale.',
        'expected': [('MOVIE', 'The Dark Knight'), ('ACTOR', 'Christian Bale')],
        'category': 'unseen_entity',  # Christian Bale is not in ACTOR_NAMES
    },
    {
        'text': "Quentin Tarantino's Pulp Fiction is a cult classic starring Samuel L. Jackson.",
        'expected': [('DIRECTOR', 'Quentin Tarantino'), ('MOVIE', 'Pulp Fiction'),
                     ('ACTOR', 'Samuel L. Jackson')],
        'category': 'in_gazetteer',
    },
    {
        'text': "I didn't enjoy Avatar despite its Golden Globe nominations.",
        'expected': [('MOVIE', 'Avatar'), ('AWARD', 'Golden Globe')],
        'category': 'in_gazetteer',
    },
    {
        'text': 'Martin Scorsese finally won an Academy Award for The Departed.',
        'expected': [('DIRECTOR', 'Martin Scorsese'), ('AWARD', 'Academy Award'),
                     ('MOVIE', 'The Departed')],
        'category': 'unseen_entity',  # The Departed is not in MOVIE_TITLES
    },
    {
        'text': 'The cinematography was gorgeous but the plot dragged in the second half.',
        'expected': [],
        'category': 'negative',
    },
    {
        'text': 'She collects Oscars memorabilia and admires Avatar-like visual design.',
        'expected': [],
        'category': 'substring_trap',
    },
    {
        'text': 'Denis Villeneuve and Greta Gerwig both deserve a BAFTA for Inception.',
        'expected': [('DIRECTOR', 'Denis Villeneuve'), ('DIRECTOR', 'Greta Gerwig'),
                     ('AWARD', 'BAFTA'), ('MOVIE', 'Inception')],
        'category': 'in_gazetteer',
    },
]


def subset(category: str) -> list[dict]:
    """Filter the test set to one category, for per-category reporting."""
    return [example for example in TEST_REVIEWS if example['category'] == category]
