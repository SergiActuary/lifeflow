from pathlib import Path

import numpy as np
import polars as pl

from lifeflow import Inforce, decrements

DATA = Path(__file__).parent / "datatest"

df = pl.read_excel(DATA / "exit_by_cf.xlsx")

qx = df["qx"].to_numpy()
wx = df["wx"].to_numpy()
exit_by_qx_ref = df["exit_by_qx"].to_numpy()
exit_by_wx_ref = df["exit_by_wx"].to_numpy()


def test_exit_by():
    dec = decrements(qx=qx, wx=wx)
    inf = Inforce(dec)
    assert np.allclose(inf.exit_by("qx"), exit_by_qx_ref, rtol=1e-10)
    assert np.allclose(inf.exit_by("wx"), exit_by_wx_ref, rtol=1e-10)
