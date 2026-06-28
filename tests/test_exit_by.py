from pathlib import Path

import numpy as np
import polars as pl

from lifeflow import Inforce, Decrement, Portfolio, Timeline

DATA = Path(__file__).parent / "datatest"


def _setup():
    df = pl.read_excel(DATA / "exit_by_cf.xlsx")
    qx = df["qx"].to_numpy()
    wx = df["wx"].to_numpy()
    T = len(qx)

    portfolio = Portfolio(pl.DataFrame({"duration": [T]}))
    tl = Timeline("duration")
    qx_dec = Decrement(qx, tl)
    wx_dec = Decrement(wx, tl)
    return portfolio, qx_dec, wx_dec, df


def test_exit_by():
    portfolio, qx_dec, wx_dec, df = _setup()
    exit_qx_ref = df["exit_by_qx"].to_numpy()
    exit_wx_ref = df["exit_by_wx"].to_numpy()

    inf = Inforce(portfolio, [qx_dec, wx_dec])

    assert np.allclose(inf.exit_by(qx_dec)[0], exit_qx_ref, rtol=1e-5), (
        f"exit_qx diferencias:\n{inf.exit_by(qx_dec)[0] - exit_qx_ref}"
    )
    assert np.allclose(inf.exit_by(wx_dec)[0], exit_wx_ref, rtol=1e-5), (
        f"exit_wx diferencias:\n{inf.exit_by(wx_dec)[0] - exit_wx_ref}"
    )
