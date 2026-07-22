"""NLP homework (Week 1, Day 1) implemented as a tested, importable package.

Covers the three tasks from ``nlp_homework.md``:

* Task 1 — :mod:`nlp_homework.preprocessing`
* Task 2 — :mod:`nlp_homework.exploration`
* Task 3 — :mod:`nlp_homework.ner`

:mod:`nlp_homework.pipeline` runs all three end to end.
"""

from __future__ import annotations

from . import dataset, exploration, ner, pipeline, preprocessing, vocabulary
from .ner import Entity
from .preprocessing import ProcessedText

__all__ = [
    'dataset', 'exploration', 'ner', 'pipeline', 'preprocessing', 'vocabulary',
    'Entity', 'ProcessedText',
]
__version__ = '1.0.0'
