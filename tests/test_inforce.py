from pathlib import Path

import numpy as np
import polars as pl

from lifeflow import Inforce, Hypothesis, Portfolio, Timeline

DATA = Path(__file__).parent / "datatest"


def _setup():
    df = pl.read_excel(DATA / "inforce_test.xlsx")
    qx = df["qx"].to_numpy()
    wx = df["wx"].to_numpy()
    T = len(qx)

    portfolio = Portfolio(pl.DataFrame({"duration": [T]}))
    tl = Timeline("duration")
    qx_hyp = Hypothesis(qx, tl)
    wx_hyp = Hypothesis(wx, tl)
    return portfolio, qx_hyp, wx_hyp, df


def test_inforce():
    portfolio, qx_hyp, wx_hyp, df = _setup()
    inf_ref = df["inf"].to_numpy()

    inforce = Inforce(portfolio, [qx_hyp, wx_hyp]).inf[0]

    assert np.allclose(inforce, inf_ref, rtol=1e-10), (
        f"Diferencias:\n{inforce - inf_ref}"
    )
