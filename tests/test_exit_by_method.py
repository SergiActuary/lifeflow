from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lifeflow import Inforce, Decrement, Portfolio, Timeline

DATA = Path(__file__).parent / "datatest"


def _setup():
    df = pl.read_excel(DATA / "exit_by_udd.xlsx")
    qx = df["qx"].to_numpy()
    wx = df["wx"].to_numpy()
    T = len(qx)

    portfolio = Portfolio(pl.DataFrame({"duration": [T]}))
    tl = Timeline("duration")
    qx_dec = Decrement(qx, tl)
    wx_dec = Decrement(wx, tl)
    return portfolio, qx_dec, wx_dec


def test_exit_by_rejects_unknown_method():
    portfolio, qx_dec, wx_dec = _setup()
    inf = Inforce(portfolio, [qx_dec, wx_dec])

    with pytest.raises(ValueError, match="method"):
        inf.exit_by(qx_dec, method="typo")


def test_exit_by_methods_differ():
    portfolio, qx_dec, wx_dec = _setup()

    udd = Inforce(portfolio, [qx_dec, wx_dec]).exit_by(qx_dec, method="udd")
    cf = Inforce(portfolio, [qx_dec, wx_dec]).exit_by(qx_dec, method="cf")

    assert not np.allclose(udd, cf, rtol=1e-12, atol=0), (
        "udd y cf devuelven lo mismo: el despacho de method no funciona"
    )


def test_exit_by_cache_switches_method():
    portfolio, qx_dec, wx_dec = _setup()

    udd_ref = Inforce(portfolio, [qx_dec, wx_dec]).exit_by(qx_dec, method="udd")
    cf_ref = Inforce(portfolio, [qx_dec, wx_dec]).exit_by(qx_dec, method="cf")

    inf = Inforce(portfolio, [qx_dec, wx_dec])

    for method, esperado in [("udd", udd_ref), ("cf", cf_ref), ("udd", udd_ref)]:
        obtenido = inf.exit_by(qx_dec, method=method)
        assert np.array_equal(obtenido, esperado), (
            f"tras pedir method={method!r} el cache devolvio otro metodo"
        )
