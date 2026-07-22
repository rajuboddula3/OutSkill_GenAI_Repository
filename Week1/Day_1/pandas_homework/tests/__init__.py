"""Test suite for :mod:`pandas_homework`.

This package marker is load-bearing. The sibling ``Day_1/tests/`` directory (the
NLP homework suite) already contains ``test_dataset.py``, ``test_exploration.py``
and ``test_pipeline.py``. Without an ``__init__.py`` here, pytest imports both
sets of files under bare top-level names and a plain ``pytest`` run from Day_1
aborts with "import file mismatch". Being a package qualifies these modules as
``pandas_homework.tests.*``, so the two suites coexist.
"""
