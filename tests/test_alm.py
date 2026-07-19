from pathlib import Path

from lifeflow import duration_hicks, duration_macaulay, convexity

import numpy as np
import polars as pl

DATA = Path(__file__).parent / "datatest"

def _setup():
    data = pl.read_excel(DATA / "alm.xlsx", sheet_name="durations")
    reference = pl.read_excel(DATA / "alm.xlsx", sheet_name="results")

    pay = data["flow"].to_numpy()
    rate = data["rate"].to_numpy()

    return pay, rate, reference

def test_alm():
    pay, rate, reference = _setup()
    macaulay = duration_macaulay(pay, rate)
    hicks = duration_hicks(pay, rate)
    convex = convexity(pay, rate)

    assert np.allclose(macaulay, reference["duration_macaulay"][0], rtol=1e-12, atol=0), (
        f"duration macaulay diff:\n{macaulay - reference["duration_macaulay"][0]}"
    )
    assert np.allclose(hicks, reference["duration_hicks"][0], rtol=1e-12, atol=0), (
        f"duration hicks diff:\n{hicks - reference["duration_hicks"][0]}"
    )
    assert np.allclose(convex, reference["convexity"][0], rtol=1e-12, atol=0), (
        f"convexity diff:\n{convex - reference["convexity"][0]}"
    )
