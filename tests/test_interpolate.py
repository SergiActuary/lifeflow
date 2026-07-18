from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lifeflow.interpolate import interpolate

DATA = Path(__file__).parent / "datatest"

SHEETS = {"linear": "udd", "geomdecr": "cf", "geomincr": "compound"}


def _load(method):
    df = pl.read_excel(DATA / "interpolate.xlsx", sheet_name=SHEETS[method])
    anual = df["anual"].drop_nulls().to_numpy()
    mensual = df["mensual"].drop_nulls().to_numpy()
    return anual, mensual


def test_interpolate_geomdecr_down():
    anual, esperado = _load("geomdecr")
    resultado = interpolate(anual, method="geomdecr", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_geomdecr_up():
    anual, mensual = _load("geomdecr")
    resultado = interpolate(mensual, method="geomdecr", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )


def test_interpolate_linear_down():
    anual, esperado = _load("linear")
    resultado = interpolate(anual, method="linear", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_linear_up():
    anual, mensual = _load("linear")
    resultado = interpolate(mensual, method="linear", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )


def test_interpolate_geomincr_down():
    anual, esperado = _load("geomincr")
    resultado = interpolate(anual, method="geomincr", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_geomincr_up():
    anual, mensual = _load("geomincr")
    resultado = interpolate(mensual, method="geomincr", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )


def test_interpolate_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        interpolate([0.1], method="udd")


def test_interpolate_rejects_unknown_direction():
    with pytest.raises(ValueError, match="to"):
        interpolate([0.1], method="linear", to="sideways")
