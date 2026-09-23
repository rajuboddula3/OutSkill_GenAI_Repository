# `nlp_homework` — Week 1 Day 1 NLP assignment

Implementation of the three tasks in [`../nlp_homework.md`](../nlp_homework.md),
packaged as an importable, tested module rather than a single-file notebook.
[`../nlp_homework_solution.ipynb`](../nlp_homework_solution.ipynb) is a thin
presentation layer over this package.

## Layout

| Module | Responsibility |
|---|---|
| `vocabulary.py` | Sentiment word lists and entity gazetteers, shared by the generator and the NER so they cannot drift apart |
| `dataset.py` | Reproducible synthetic corpus generation, CSV persistence |
| `preprocessing.py` | Task 1 — tokenising, stopword removal, stemming, POS-aware lemmatisation |
| `exploration.py` | Task 2 — statistics, n-grams, TF-IDF, figures |
| `ner.py` | Task 3 — NLTK + gazetteer NER, overlap resolution, evaluation, HTML highlighting |
| `evaluation_data.py` | Manually labelled NER test set, including negatives and unseen entities |
| `pipeline.py` | End-to-end orchestration; returns data, prints nothing |

## Usage

```python
from nlp_homework import pipeline

result = pipeline.run(output_dir='.', n_reviews=1000, seed=42)
print(result.task_two.statistics)
print(result.task_three.overall_metrics)
```

Individual stages are usable on their own:

```python
from nlp_homework import ner, preprocessing

preprocessing.preprocess("The films were absolutely brilliant!").lemmatized
ner.custom_movie_ner("Steven Spielberg's Jurassic Park won an Oscar.")
```

## Tests

From the `Week1/` directory:

```bash
uv run pytest Day_1/tests -v
```

118 tests, no network required — the suite substitutes in-repo filler text for
the 20-newsgroups fetch. Tests marked `REGRESSION` in their docstring pin a
specific defect from the original notebook implementation.

## Design notes

**Entities carry character offsets.** `Entity(start, end, label, text)` is
produced by both recognisers and threaded through to rendering. Re-deriving
positions with `str.find` at render time causes three distinct defects: only the
first mention of a term is found, substrings match inside longer words, and
overlapping spans get emitted twice.

**Overlap resolution is centralised.** `resolve_overlaps` keeps the longest span
and prefers domain labels over generic ones, so NLTK's `PERSON` yields to
`DIRECTOR` for the same text.

**Lemmatisation is POS-aware.** `WordNetLemmatizer` defaults to noun; untagged,
it leaves `running` and `better` unchanged, which makes it barely distinguishable
from doing nothing.

**Narrative text is derived, not hardcoded.** `describe_comparison` builds the
stemming-vs-lemmatisation discussion from the computed table, so the prose cannot
contradict the numbers printed above it.

**The NER test set is adversarial.** It contains entities outside the gazetteer,
reviews with no entities, and substring traps (`Oscars`, `Avatar-like`).
Evaluating a dictionary recogniser only on its own dictionary yields a
meaningless F1 = 1.00.

## Current NER performance

Against `evaluation_data.TEST_REVIEWS`:

| Slice | Precision | Recall | F1 |
|---|---|---|---|
| Overall | 1.00 | 0.88 | 0.94 |
| In-gazetteer | 1.00 | 1.00 | 1.00 |
| Unseen entities | 1.00 | 0.60 | 0.75 |
| Negatives / traps | no false positives | — | — |

Recall is limited by gazetteer coverage, not by the matching algorithm — every
miss is an entity the dictionary does not contain. Closing that gap requires a
statistical or transformer-based model, not a longer word list.
