"""Task 2 — corpus statistics, n-grams, TF-IDF and figures.

Plotting is kept separate from computation so every statistic is testable
without a display backend.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.use('Agg')  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from wordcloud import WordCloud  # noqa: E402


def corpus_statistics(token_lists: pd.Series) -> dict[str, float]:
    """Average / median / min / max review length plus vocabulary size."""
    lengths = token_lists.apply(len)
    vocabulary = {token for tokens in token_lists for token in tokens}
    return {
        'n_documents': int(len(token_lists)),
        'average_length': float(lengths.mean()),
        'median_length': float(lengths.median()),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'vocabulary_size': len(vocabulary),
        'total_tokens': int(lengths.sum()),
    }


def most_common_words(token_lists: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    return Counter(t for tokens in token_lists for t in tokens).most_common(top_n)


def ngram_frequencies(token_lists: pd.Series, n: int, top_n: int = 15) -> list[tuple[str, int]]:
    """Top ``top_n`` n-grams, joined into readable strings.

    N-grams are counted per document so they never span a document boundary.
    """
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(ngrams(tokens, n))
    return [(' '.join(gram), count) for gram, count in counter.most_common(top_n)]


def top_tfidf_terms(
    documents: pd.Series,
    labels: pd.Series,
    target_label: int,
    top_n: int = 20,
    max_features: int = 1000,
) -> list[tuple[str, float]]:
    """Mean TF-IDF per term across documents carrying ``target_label``.

    No ``stop_words`` filter here: the input is already stopword-free, and
    re-filtering would silently mask a preprocessing regression.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(documents)
    features = vectorizer.get_feature_names_out()

    mask = np.asarray(labels == target_label)
    if not mask.any():
        return []

    means = np.asarray(matrix[mask].mean(axis=0)).ravel()
    top_indices = np.argsort(means)[::-1][:top_n]
    return [(features[i], float(means[i])) for i in top_indices]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_length_distribution(token_lists: pd.Series, output_path: Path) -> Path:
    lengths = token_lists.apply(len)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=30, kde=True, ax=ax)
    ax.set(title='Distribution of Review Lengths', xlabel='Number of Words', ylabel='Frequency')
    return _save(fig, output_path)


def plot_wordcloud(tokens: list[str], title: str, output_path: Path) -> Path | None:
    if not tokens:
        return None
    cloud = WordCloud(width=800, height=400, background_color='white',
                      max_words=200).generate(' '.join(tokens))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return _save(fig, output_path)


def plot_ranked_bars(
    panels: list[tuple[str, list[tuple[str, float]]]],
    output_path: Path,
    xlabel: str = 'Frequency',
) -> Path:
    """Stacked horizontal bar panels, highest value at the top of each panel."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 5 * len(panels)), squeeze=False)
    for ax, (title, items) in zip(axes.ravel(), panels):
        labels = [label for label, _ in items][::-1]
        values = [value for _, value in items][::-1]
        ax.barh(range(len(labels)), values, align='center')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set(title=title, xlabel=xlabel)
    return _save(fig, output_path)


def plot_counts(counts: pd.Series, title: str, output_path: Path,
                xlabel: str = '', ylabel: str = 'Count') -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return _save(fig, output_path)


def plot_grouped_counts(frame: pd.DataFrame, title: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    frame.plot(kind='bar', ax=ax)
    ax.set(title=title, xlabel='Entity Type', ylabel='Count')
    ax.legend()
    return _save(fig, output_path)


def _save(fig, output_path: Path) -> Path:
    """Write a figure and close it, so long runs do not leak figure handles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path
