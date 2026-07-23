from pathlib import Path

import numpy as np
import polars as pl

from lifeflow import Probabilities, Decrement, Portfolio, Timeline

DATA = Path(__file__).parent / "datatest"


def _setup():
    df = pl.read_excel(DATA / "inforce_test.xlsx")
    qx = df["qx"].to_numpy()
    wx = df["wx"].to_numpy()
    T = len(qx)

    portfolio = Portfolio(pl.DataFrame({"duration": [T]}))
    tl = Timeline("duration")
    qx_dec = Decrement(qx, tl)
    wx_dec = Decrement(wx, tl)
    return portfolio, qx_dec, wx_dec, df


def test_inforce():
    portfolio, qx_dec, wx_dec, df = _setup()
    inf_ref = df["inf"].to_numpy()

    inforce = Probabilities(portfolio, [qx_dec, wx_dec]).inforce[0]

    assert np.allclose(inforce, inf_ref, rtol=1e-10), (
        f"Diferencias:\n{inforce - inf_ref}"
    )
