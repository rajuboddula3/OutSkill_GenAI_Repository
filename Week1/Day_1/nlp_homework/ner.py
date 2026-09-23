"""Task 3 — named entity recognition, evaluation and highlighting.

Entities carry character offsets throughout. The naive implementation passed
``(label, text)`` pairs around and re-derived positions with ``str.find`` at
render time, which caused three separate defects: only the first mention of a
term was ever found, substrings matched inside longer words, and overlapping
spans were emitted twice into the HTML.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize

from .vocabulary import GAZETTEERS

#: Rendering colours per label. Unknown labels fall back to light grey.
ENTITY_COLORS: dict[str, str] = {
    'PERSON': '#ffadad',
    'ORGANIZATION': '#ffd6a5',
    'LOCATION': '#caffbf',
    'GPE': '#caffbf',
    'GSP': '#caffbf',
    'FACILITY': '#a0c4ff',
    'DATE': '#e2e2e2',
    'DIRECTOR': '#9bf6ff',
    'ACTOR': '#bdb2ff',
    'MOVIE': '#ffc6ff',
    'AWARD': '#fdffb6',
    'POTENTIAL_MOVIE': '#fffffc',
}
_DEFAULT_COLOR = '#e2e2e2'

#: Labels from the domain gazetteers outrank NLTK's generic labels when both
#: cover the same span — DIRECTOR is strictly more informative than PERSON.
_LABEL_PRIORITY: dict[str, int] = {
    'DIRECTOR': 0, 'ACTOR': 0, 'MOVIE': 0, 'AWARD': 0,
    'POTENTIAL_MOVIE': 1,
}
_GENERIC_PRIORITY = 2

_QUOTED_TITLE_RE = re.compile(r'"([A-Z][^"]+)"')


@dataclass(frozen=True, order=True)
class Entity:
    """One entity mention, anchored to a character span in the source text."""

    start: int
    end: int
    label: str
    text: str

    def overlaps(self, other: 'Entity') -> bool:
        return self.start < other.end and other.start < self.end

    def as_pair(self) -> tuple[str, str]:
        """``(label, text)`` — the comparison key used by :func:`evaluate`."""
        return (self.label, self.text)


@lru_cache(maxsize=None)
def _gazetteer_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile one alternation matching any term at a word boundary.

    Terms are sorted longest-first so 'Academy Award' wins over a bare 'Award'.
    The boundary class includes ``-`` as well as ``\\w`` so neither 'Oscars' nor
    'Avatar-like' registers a hit; hyphens *inside* a term ('Daniel Day-Lewis')
    are unaffected because they sit within the alternation, not at its edge.
    """
    ordered = sorted(terms, key=len, reverse=True)
    alternation = '|'.join(re.escape(term) for term in ordered)
    return re.compile(rf'(?<![\w-])(?:{alternation})(?![\w-])', re.IGNORECASE)


def extract_entities_nltk(text: str) -> list[Entity]:
    """Extract entities with NLTK's ``ne_chunk``, recovering character offsets.

    ``ne_chunk`` yields tokens, not offsets, so each chunk is re-located in the
    source by scanning forward from the previous match. Scanning forward (rather
    than searching from position 0) is what keeps repeated mentions distinct.
    """
    tokens = word_tokenize(text)
    chunks = ne_chunk(pos_tag(tokens))

    entities: list[Entity] = []
    cursor = 0
    for chunk in chunks:
        if not hasattr(chunk, 'label'):
            # A plain (token, tag) leaf — advance the cursor past it.
            cursor = _advance(text, chunk[0], cursor)
            continue

        words = [token for token, _ in chunk]
        start = _advance(text, words[0], cursor, return_start=True)
        if start is None:
            continue
        end = start
        for word in words:
            found = _advance(text, word, end, return_start=True)
            if found is None:
                break
            end = found + len(word)
        entities.append(Entity(start, end, chunk.label(), text[start:end]))
        cursor = end
    return entities


def _advance(text: str, token: str, cursor: int, return_start: bool = False):
    """Locate ``token`` at or after ``cursor``; return its end (or start)."""
    index = text.find(token, cursor)
    if index == -1:
        return None if return_start else cursor
    return index if return_start else index + len(token)


def custom_movie_ner(text: str) -> list[Entity]:
    """Dictionary + pattern NER for movie-domain entities.

    Finds *every* mention of every term, not just the first, and requires word
    boundaries so 'Avatar' does not match inside 'Avatar-like'.
    """
    entities: list[Entity] = []

    for label, terms in GAZETTEERS.items():
        for match in _gazetteer_pattern(terms).finditer(text):
            entities.append(Entity(match.start(), match.end(), label, match.group()))

    # Quoted capitalised phrases are candidate titles the gazetteer misses.
    for match in _QUOTED_TITLE_RE.finditer(text):
        start, end = match.span(1)
        candidate = Entity(start, end, 'POTENTIAL_MOVIE', match.group(1))
        if not any(e.overlaps(candidate) for e in entities):
            entities.append(candidate)

    return sorted(entities)


def resolve_overlaps(entities: list[Entity]) -> list[Entity]:
    """Drop overlapping mentions, keeping the most specific and longest span.

    Sort key: earliest start, then longest span, then gazetteer labels ahead of
    NLTK's generic ones. Without this the highlighter emitted the same substring
    once per overlapping label, duplicating text in the output.
    """
    ordered = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), _LABEL_PRIORITY.get(e.label, _GENERIC_PRIORITY)),
    )
    kept: list[Entity] = []
    for entity in ordered:
        if kept and entity.overlaps(kept[-1]):
            continue
        kept.append(entity)
    return kept


def extract_all_entities(text: str) -> list[Entity]:
    """Combine the NLTK and custom extractors into one non-overlapping list."""
    return resolve_overlaps(extract_entities_nltk(text) + custom_movie_ner(text))


def highlight_entities(text: str, entities: list[Entity]) -> str:
    """Render ``text`` as HTML with each entity span colour-coded.

    Both the surrounding text and the entity text are escaped, so reviews
    containing ``&`` or ``<b>`` render literally instead of as markup.
    """
    parts: list[str] = []
    cursor = 0
    for entity in resolve_overlaps(entities):
        if entity.start < cursor:
            continue
        parts.append(html.escape(text[cursor:entity.start]))
        color = ENTITY_COLORS.get(entity.label, _DEFAULT_COLOR)
        parts.append(
            f'<span style="background-color: {color};" '
            f'title="{html.escape(entity.label, quote=True)}">'
            f'{html.escape(text[entity.start:entity.end])}</span>'
        )
        cursor = entity.end
    parts.append(html.escape(text[cursor:]))
    return ''.join(parts)


def count_by_label(entities: list[Entity]) -> Counter:
    return Counter(entity.label for entity in entities)


def evaluate(test_data: list[dict], ner_function) -> dict[str, float]:
    """Micro-averaged precision / recall / F1 over ``(label, text)`` pairs.

    ``test_data`` items are ``{"text": str, "expected": [(label, text), ...]}``.
    An empty ``expected`` list is a valid negative example: any prediction on it
    counts as a false positive.
    """
    tp = fp = fn = 0
    for example in test_data:
        expected = {tuple(pair) for pair in example['expected']}
        predicted = {e.as_pair() for e in ner_function(example['text'])}
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
    }


def evaluate_by_label(test_data: list[dict], ner_function) -> dict[str, dict[str, float]]:
    """Per-label metrics, so a strong average cannot mask one broken category."""
    labels = {label for ex in test_data for label, _ in ex['expected']}
    results = {}
    for label in sorted(labels):
        subset = [
            {'text': ex['text'],
             'expected': [p for p in ex['expected'] if p[0] == label]}
            for ex in test_data
        ]
        results[label] = evaluate(
            subset,
            lambda t, lbl=label: [e for e in ner_function(t) if e.label == lbl],
        )
    return results
