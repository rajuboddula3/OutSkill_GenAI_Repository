"""Pandas homework (Week 1, Day 1) implemented as a tested, importable package.

Covers the five tasks from ``pandas_homework.md``:

* Task 1 — :mod:`pandas_homework.exploration`
* Task 2 — :mod:`pandas_homework.filtering`
* Task 3 — :mod:`pandas_homework.transformation`
* Task 4 — :mod:`pandas_homework.aggregation`
* Task 5 — :mod:`pandas_homework.visualization`

:mod:`pandas_homework.pipeline` runs all five end to end.
"""

from __future__ import annotations

from . import (
    aggregation,
    dataset,
    exploration,
    filtering,
    pipeline,
    transformation,
    visualization,
)
from .exploration import Exploration
from .pipeline import PipelineResult

__all__ = [
    'aggregation', 'dataset', 'exploration', 'filtering', 'pipeline',
    'transformation', 'visualization',
    'Exploration', 'PipelineResult',
]
__version__ = '1.0.0'
