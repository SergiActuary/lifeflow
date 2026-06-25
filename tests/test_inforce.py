from pathlib import Path

import numpy as np
import polars as pl

from lifeflow import Inforce, decrements

DATA = Path(__file__).parent / "datatest"

df = pl.read_excel(DATA / "inforce_test.xlsx")

qx = df["qx"].to_numpy()
wx = df["wx"].to_numpy()
inf_reference = df["inf"].to_numpy()


def test_inforce():
    dec = decrements(qx=qx, wx=wx)
    inf = Inforce(dec)
    assert np.allclose(inf.inforce, inf_reference, rtol=1e-10)
