from pathlib import Path

import numpy as np
import polars as pl

from lifeflow.interpolate import interpolate

DATA = Path(__file__).parent / "datatest"


def _load(sheet):
    df = pl.read_excel(DATA / "interpolate.xlsx", sheet_name=sheet)
    anual = df["anual"].drop_nulls().to_numpy()
    mensual = df["mensual"].drop_nulls().to_numpy()
    return anual, mensual


def test_interpolate_cf_down():
    anual, esperado = _load("cf")
    resultado = interpolate(anual, method="cf", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_cf_up():
    anual, mensual = _load("cf")
    resultado = interpolate(mensual, method="cf", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )


def test_interpolate_udd_down():
    anual, esperado = _load("udd")
    resultado = interpolate(anual, method="udd", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_udd_up():
    anual, mensual = _load("udd")
    resultado = interpolate(mensual, method="udd", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )


def test_interpolate_compound_down():
    anual, esperado = _load("compound")
    resultado = interpolate(anual, method="compound", to="down")
    assert np.allclose(resultado, esperado, rtol=1e-10), (
        f"Diferencias:\n{resultado - esperado}"
    )


def test_interpolate_compound_up():
    anual, mensual = _load("compound")
    resultado = interpolate(mensual, method="compound", to="up")
    assert np.allclose(resultado, anual, rtol=1e-10), (
        f"Diferencias:\n{resultado - anual}"
    )
