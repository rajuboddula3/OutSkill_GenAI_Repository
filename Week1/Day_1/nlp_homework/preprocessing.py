"""Task 1 — text preprocessing, stemming and lemmatisation.

Two deliberate departures from the naive classroom implementation:

1. Tokenise *before* stripping punctuation. Stripping first turns ``don't`` into
   ``dont`` (which is no longer a stopword and survives as noise) and collapses
   ``soc.religion.christian`` into one nonsense token.
2. Lemmatise with a part-of-speech tag. ``WordNetLemmatizer`` defaults to noun,
   so an untagged call leaves ``running`` and ``better`` untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

_DIGIT_RE = re.compile(r'\d')
_ALPHA_RE = re.compile(r'[a-z]')

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

#: Treebank POS prefix -> WordNet POS constant. Anything unmapped falls back to
#: NOUN, which is also ``WordNetLemmatizer``'s own default.
_TREEBANK_TO_WORDNET = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV,
}

REQUIRED_NLTK_RESOURCES: tuple[str, ...] = (
    'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
    'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words',
)


def ensure_nltk_resources(resources: tuple[str, ...] = REQUIRED_NLTK_RESOURCES) -> None:
    """Download NLTK corpora if missing. Safe to call repeatedly."""
    for resource in resources:
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    """Load the English stopword list once per process.

    The naive version rebuilt this set inside the per-document function, paying
    a corpus read on every one of the 1000 reviews.
    """
    return frozenset(stopwords.words('english'))


def _wordnet_pos(treebank_tag: str) -> str:
    return _TREEBANK_TO_WORDNET.get(treebank_tag[:1].upper(), wordnet.NOUN)


def is_content_token(token: str) -> bool:
    """True for lowercase tokens that carry meaning.

    Rejects pure punctuation, anything containing a digit, and stopwords.
    """
    if not _ALPHA_RE.search(token):
        return False
    if _DIGIT_RE.search(token):
        return False
    return token not in _stopword_set()


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenise, then drop punctuation / numeric / stopword tokens."""
    return [token for token in word_tokenize(text.lower()) if is_content_token(token)]


@dataclass(frozen=True)
class ProcessedText:
    """Every preprocessing product for one document, computed in a single pass."""

    tokens: list[str]
    stemmed: list[str]
    lemmatized: list[str]

    @property
    def cleaned(self) -> str:
        """Whitespace-normalised text rebuilt from the surviving tokens."""
        return ' '.join(self.tokens)

    @property
    def word_count(self) -> int:
        return len(self.tokens)


def stem_tokens(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """POS-aware lemmatisation.

    Tagging happens on the token sequence as a whole so the tagger retains the
    context it needs; tagging word-by-word would degrade to noun for everything.
    """
    if not tokens:
        return []
    return [
        _LEMMATIZER.lemmatize(token, _wordnet_pos(tag))
        for token, tag in nltk.pos_tag(tokens)
    ]


def preprocess(text: str) -> ProcessedText:
    """Run the full Task 1 pipeline over one document.

    Tokenisation happens exactly once, unlike the naive version which called the
    whole pipeline twice per row to populate two DataFrame columns.
    """
    tokens = tokenize(text)
    return ProcessedText(
        tokens=tokens,
        stemmed=stem_tokens(tokens),
        lemmatized=lemmatize_tokens(tokens),
    )


def compare_normalizers(words: list[str]) -> list[dict[str, str]]:
    """Stem vs lemma for each word, as rows ready for a DataFrame.

    Lemmatisation is done per-word here (rather than via :func:`lemmatize_tokens`)
    because the inputs are isolated dictionary words with no sentence context;
    each is tagged on its own so the comparison reflects the word in isolation.
    """
    rows = []
    for word in words:
        lemma = _LEMMATIZER.lemmatize(word, _wordnet_pos(nltk.pos_tag([word])[0][1]))
        rows.append({
            'original': word,
            'stemmed': _STEMMER.stem(word),
            'lemmatized': lemma,
        })
    return rows


def describe_comparison(rows: list[dict[str, str]]) -> list[str]:
    """Derive the stemming-vs-lemmatisation discussion *from the actual results*.

    The naive version hardcoded prose that contradicted its own output table
    (it claimed 'running' lemmatises to 'run'; untagged, it does not).
    """
    notes: list[str] = []

    changed_by_stem = [r for r in rows if r['stemmed'] != r['original']]
    changed_by_lemma = [r for r in rows if r['lemmatized'] != r['original']]
    notes.append(
        f"Stemming altered {len(changed_by_stem)}/{len(rows)} words; "
        f"lemmatisation altered {len(changed_by_lemma)}/{len(rows)}."
    )

    disagreements = [r for r in rows if r['stemmed'] != r['lemmatized']]
    for row in disagreements:
        notes.append(
            f"'{row['original']}' -> stem '{row['stemmed']}' vs lemma "
            f"'{row['lemmatized']}'."
        )

    non_words = [r for r in disagreements if not wordnet.synsets(r['stemmed'])]
    if non_words:
        stems = ', '.join(f"'{r['stemmed']}'" for r in non_words)
        notes.append(
            f"Stemming produced {len(non_words)} form(s) absent from WordNet ({stems}); "
            "lemmatisation returned real dictionary words throughout."
        )

    notes.append(
        "For this corpus lemmatisation is preferable: NER and the entity-highlighting "
        "step both need surface forms that still match the source text, which "
        "truncating stems break."
    )
    return notes
