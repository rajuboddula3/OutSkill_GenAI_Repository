"""End-to-end orchestration of Tasks 1-3.

Every stage returns data; nothing prints. The notebook (or ``__main__`` below)
is responsible for presentation, which keeps the pipeline testable headlessly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import exploration, ner, preprocessing
from .dataset import generate_dataset, save_dataset
from .evaluation_data import TEST_REVIEWS

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_WORDS = ['running', 'better', 'studies', 'movies', 'directed', 'worst']


@dataclass
class TaskOneResult:
    frame: pd.DataFrame
    comparison: list[dict[str, str]]
    discussion: list[str]


@dataclass
class TaskTwoResult:
    statistics: dict[str, float]
    positive_words: list[tuple[str, int]]
    negative_words: list[tuple[str, int]]
    positive_bigrams: list[tuple[str, int]]
    negative_bigrams: list[tuple[str, int]]
    positive_trigrams: list[tuple[str, int]]
    negative_trigrams: list[tuple[str, int]]
    positive_tfidf: list[tuple[str, float]]
    negative_tfidf: list[tuple[str, float]]
    figures: list[Path] = field(default_factory=list)


@dataclass
class TaskThreeResult:
    entities: pd.DataFrame
    custom_entities: pd.DataFrame
    overall_metrics: dict[str, float]
    per_label_metrics: dict[str, dict[str, float]]
    category_metrics: dict[str, dict[str, float]]
    highlighted_samples: list[tuple[int, str]]
    figures: list[Path] = field(default_factory=list)


def run_task_one(frame: pd.DataFrame,
                 comparison_words: list[str] | None = None) -> TaskOneResult:
    """Preprocess every review once, then compare stemming with lemmatisation."""
    processed = frame['review'].apply(preprocessing.preprocess)
    enriched = frame.assign(
        tokens=[p.tokens for p in processed],
        preprocessed=[p.cleaned for p in processed],
        stemmed=[p.stemmed for p in processed],
        lemmatized=[p.lemmatized for p in processed],
        word_count=[p.word_count for p in processed],
    )
    rows = preprocessing.compare_normalizers(comparison_words or DEFAULT_COMPARISON_WORDS)
    return TaskOneResult(enriched, rows, preprocessing.describe_comparison(rows))


def run_task_two(frame: pd.DataFrame, output_dir: Path) -> TaskTwoResult:
    """Corpus statistics, common words, n-grams, TF-IDF and their figures."""
    output_dir = Path(output_dir)
    positive = frame[frame['sentiment'] == 1]
    negative = frame[frame['sentiment'] == 0]
    positive_tokens = [t for tokens in positive['tokens'] for t in tokens]
    negative_tokens = [t for tokens in negative['tokens'] for t in tokens]

    result = TaskTwoResult(
        statistics=exploration.corpus_statistics(frame['tokens']),
        positive_words=exploration.most_common_words(positive['tokens']),
        negative_words=exploration.most_common_words(negative['tokens']),
        positive_bigrams=exploration.ngram_frequencies(positive['tokens'], 2),
        negative_bigrams=exploration.ngram_frequencies(negative['tokens'], 2),
        positive_trigrams=exploration.ngram_frequencies(positive['tokens'], 3),
        negative_trigrams=exploration.ngram_frequencies(negative['tokens'], 3),
        positive_tfidf=exploration.top_tfidf_terms(frame['preprocessed'], frame['sentiment'], 1),
        negative_tfidf=exploration.top_tfidf_terms(frame['preprocessed'], frame['sentiment'], 0),
    )

    figures = [
        exploration.plot_length_distribution(frame['tokens'],
                                             output_dir / 'review_length_distribution.png'),
        exploration.plot_wordcloud(positive_tokens, 'Word Cloud - Positive Reviews',
                                   output_dir / 'positive_wordcloud.png'),
        exploration.plot_wordcloud(negative_tokens, 'Word Cloud - Negative Reviews',
                                   output_dir / 'negative_wordcloud.png'),
        exploration.plot_ranked_bars(
            [('Top Bigrams - Positive Reviews', result.positive_bigrams),
             ('Top Bigrams - Negative Reviews', result.negative_bigrams)],
            output_dir / 'bigram_frequencies.png'),
        exploration.plot_ranked_bars(
            [('Top Trigrams - Positive Reviews', result.positive_trigrams),
             ('Top Trigrams - Negative Reviews', result.negative_trigrams)],
            output_dir / 'trigram_frequencies.png'),
        exploration.plot_ranked_bars(
            [('Top 20 TF-IDF Terms - Positive Reviews', result.positive_tfidf),
             ('Top 20 TF-IDF Terms - Negative Reviews', result.negative_tfidf)],
            output_dir / 'tfidf_scores.png', xlabel='Mean TF-IDF Score'),
    ]
    result.figures = [f for f in figures if f is not None]
    return result


def _entity_frame(frame: pd.DataFrame, extractor) -> pd.DataFrame:
    """Run ``extractor`` over every review, flattening mentions into rows."""
    records = [
        {'review_id': idx, 'sentiment': row['sentiment'], 'entity_type': entity.label,
         'entity_text': entity.text, 'start': entity.start, 'end': entity.end}
        for idx, row in frame.iterrows()
        for entity in extractor(row['review'])
    ]
    columns = ['review_id', 'sentiment', 'entity_type', 'entity_text', 'start', 'end']
    return pd.DataFrame(records, columns=columns)


def run_task_three(frame: pd.DataFrame, output_dir: Path, sample_size: int = 50,
                   seed: int = 42, n_highlighted: int = 3) -> TaskThreeResult:
    """NLTK NER, custom domain NER, honest evaluation and HTML highlighting."""
    output_dir = Path(output_dir)
    sample = frame.sample(min(sample_size, len(frame)), random_state=seed)

    entities = _entity_frame(sample, ner.extract_entities_nltk)
    custom = _entity_frame(sample, ner.custom_movie_ner)

    figures: list[Path] = []
    if not entities.empty:
        type_counts = entities['entity_type'].value_counts()
        figures.append(exploration.plot_counts(
            type_counts, 'Distribution of Entity Types',
            output_dir / 'entity_type_distribution.png', xlabel='Entity Type'))

        panels = [
            (f'Top 10 {label} Entities',
             list(entities[entities['entity_type'] == label]['entity_text']
                  .value_counts().head(10).items()))
            for label in type_counts.index[:4]
        ]
        figures.append(exploration.plot_ranked_bars(
            panels, output_dir / 'top_entities_by_type.png'))

        comparison = pd.DataFrame({
            'Positive': entities[entities['sentiment'] == 1]['entity_type'].value_counts(),
            'Negative': entities[entities['sentiment'] == 0]['entity_type'].value_counts(),
        }).fillna(0)
        figures.append(exploration.plot_grouped_counts(
            comparison, 'Entity Types in Positive vs Negative Reviews',
            output_dir / 'entity_comparison.png'))
    else:
        logger.warning('NLTK NER found no entities in the %d-review sample', len(sample))

    if not custom.empty:
        figures.append(exploration.plot_counts(
            custom['entity_type'].value_counts(),
            'Distribution of Custom Movie Entity Types',
            output_dir / 'custom_entity_types.png', xlabel='Entity Type'))
    else:
        logger.warning('Custom NER found no entities in the %d-review sample', len(sample))

    categories = {example['category'] for example in TEST_REVIEWS}
    category_metrics = {
        category: ner.evaluate(
            [e for e in TEST_REVIEWS if e['category'] == category], ner.custom_movie_ner)
        for category in sorted(categories)
    }

    highlighted = [
        (int(idx), ner.highlight_entities(row['review'],
                                          ner.extract_all_entities(row['review'])))
        for idx, row in sample.head(n_highlighted).iterrows()
    ]

    return TaskThreeResult(
        entities=entities,
        custom_entities=custom,
        overall_metrics=ner.evaluate(TEST_REVIEWS, ner.custom_movie_ner),
        per_label_metrics=ner.evaluate_by_label(TEST_REVIEWS, ner.custom_movie_ner),
        category_metrics=category_metrics,
        highlighted_samples=highlighted,
        figures=figures,
    )


@dataclass
class PipelineResult:
    dataset: pd.DataFrame
    task_one: TaskOneResult
    task_two: TaskTwoResult
    task_three: TaskThreeResult


def run(output_dir: Path | str = '.', n_reviews: int = 1000, seed: int = 42,
        sample_size: int = 50, dataset_path: Path | str | None = None,
        download_resources: bool = True) -> PipelineResult:
    """Run the whole assignment end to end and return every artefact."""
    output_dir = Path(output_dir)
    if download_resources:
        preprocessing.ensure_nltk_resources()

    logger.info('Generating %d reviews (seed=%d)', n_reviews, seed)
    frame = generate_dataset(n_reviews=n_reviews, seed=seed)
    if dataset_path is not None:
        save_dataset(frame, dataset_path)

    task_one = run_task_one(frame)
    task_two = run_task_two(task_one.frame, output_dir)
    task_three = run_task_three(task_one.frame, output_dir, sample_size=sample_size, seed=seed)
    return PipelineResult(frame, task_one, task_two, task_three)
