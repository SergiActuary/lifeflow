"""parallel=True debe dar el mismo resultado que jit solo (y que Python puro)."""

import numpy as np
import polars as pl
import pytest

from lifeflow import Portfolio, Timeline, grid


def _setup():
    df = pl.DataFrame({
        "id": ["A", "B", "C", "D"],
        "duration_end": [3.0, 5.0, 4.0, 2.0],
        "capital": [100.0, 200.0, 150.0, 80.0],
        "growth": [0.03, 0.02, 0.05, 0.04],
    })
    return Portfolio(df, id_col="id"), Timeline("duration_end")


def test_parallel_matches_serial():
    port, tl = _setup()

    def flow(t, capital, growth):
        return capital * (1 + growth) ** t

    plain = grid(port, tl)(flow)()
    jitted = grid(port, tl, jit=True)(flow)()
    par = grid(port, tl, jit=True, parallel=True)(flow)()

    np.testing.assert_allclose(jitted, plain)
    np.testing.assert_allclose(par, plain)


def test_parallel_requires_jit():
    port, tl = _setup()
    with pytest.raises(ValueError):
        grid(port, tl, parallel=True)
