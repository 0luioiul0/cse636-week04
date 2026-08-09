"""The docstrings in scaling.py contain worked examples; keep them true."""
import doctest

import scaling


def test_scaling_docstrings():
    results = doctest.testmod(scaling, verbose=False)
    assert results.failed == 0, f"{results.failed} of {results.attempted} doctests failed"
    assert results.attempted >= 4
