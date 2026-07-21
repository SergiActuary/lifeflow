from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lifeflow import Portfolio, Timeline, extend_t, grid

DATA = Path(__file__).parent / "datatest"

# prima creciente aritmética: 400 el primer mes, +30 cada mes (ver hoja notes)
PRIMA_0 = 400.0
INCR = 30.0
DURACION = 34


def prima(t):
    return PRIMA_0 + INCR * (t - 1)


def _portfolio(duraciones):
    df = pl.DataFrame({"duration": duraciones, "capital": [1000.0] * len(duraciones)})
    return Portfolio(df), Timeline("duration")


def test_payable_post_reproduce_el_excel():
    referencia = pl.read_excel(DATA / "flow_prepay_postpay.xlsx", sheet_name="results")
    p, tl = _portfolio([DURACION])

    @grid(p, tl)
    def f(t):
        return prima(t)

    assert np.allclose(f()[0], referencia["Prima_post"].to_numpy(), rtol=1e-12, atol=0)


def test_payable_pre_reproduce_el_excel():
    """La lógica prepagable (t+1) la escribe el actuario; payable='pre' solo
    ajusta la vigencia del último pago."""
    referencia = pl.read_excel(DATA / "flow_prepay_postpay.xlsx", sheet_name="results")
    p, tl = _portfolio([DURACION])

    @grid(p, tl, payable="pre")
    def f(t):
        return prima(t + 1)

    assert np.allclose(f()[0], referencia["Prima_pre"].to_numpy(), rtol=1e-12, atol=0)


def test_payable_pre_respeta_el_vencimiento_de_cada_poliza():
    """Con plazos escalonados el cero debe caer en la última columna vigente
    de cada póliza, no al final del horizonte."""
    duraciones = [5, 12, DURACION]
    p, tl = _portfolio(duraciones)

    @grid(p, tl, payable="pre")
    def f(t):
        return prima(t + 1)

    salida = f()
    for n, d in enumerate(duraciones):
        assert np.allclose(salida[n, : d - 1], [prima(t + 1) for t in range(1, d)]), (
            f"poliza {n} (plazo {d}): importes prepagables incorrectos"
        )
        assert np.all(salida[n, d - 1 :] == 0.0), (
            f"poliza {n} (plazo {d}): pagos fuera de contrato\n{salida[n, d - 1:]}"
        )


def test_payable_desconocido_lanza():
    p, tl = _portfolio([12])
    with pytest.raises(ValueError, match="payable"):
        grid(p, tl, payable="mid")


def test_grid_cablea_t_y_poliza():
    """Sonda: cada celda codifica su t y su póliza, así que un cruce de filas
    o un desfase temporal salta de inmediato."""
    df = pl.DataFrame({"duration": [3, 3, 3], "edad": [30.0, 45.0, 60.0]})
    p, tl = Portfolio(df), Timeline("duration")

    @grid(p, tl)
    def sonda(t, edad):
        return t * 1000.0 + edad

    esperado = np.array([[t * 1000.0 + e for t in range(1, 4)] for e in [30.0, 45.0, 60.0]])
    assert np.array_equal(sonda(), esperado)


def test_grid_empieza_en_t_igual_1():
    p, tl = _portfolio([4])

    @grid(p, tl)
    def cual_t(t):
        return float(t)

    assert np.array_equal(cual_t()[0], np.array([1.0, 2.0, 3.0, 4.0]))


def test_jit_y_no_jit_coinciden():
    """Dos implementaciones independientes (numba.vectorize y bucle joblib)
    deben dar exactamente lo mismo."""
    df = pl.DataFrame({"duration": [6, 6], "capital": [100.0, 250.0]})
    p, tl = Portfolio(df), Timeline("duration")

    def formula(t, capital):
        return capital / (1.0 + 0.03) ** (t / 12) + t

    for payable in ("post", "pre"):
        lento = grid(p, tl, jit=False, payable=payable)(formula)()
        rapido = grid(p, tl, jit=True, payable=payable)(formula)()
        assert np.array_equal(lento, rapido), f"jit y no-jit difieren con payable={payable}"


def test_grid_desempaqueta_retornos_multiples():
    p, tl = _portfolio([4])

    @grid(p, tl)
    def dos(t):
        return t * 10.0, t * 100.0

    salida = dos()
    assert isinstance(salida, tuple) and len(salida) == 2
    assert np.array_equal(salida[0][0], np.array([10.0, 20.0, 30.0, 40.0]))
    assert np.array_equal(salida[1][0], np.array([100.0, 200.0, 300.0, 400.0]))


def test_extend_t_devuelve_vector_de_longitud_T():
    @extend_t(5)
    def v(t, rate):
        return 1.0 / (1.0 + rate) ** t

    salida = v(0.03)
    assert salida.shape == (5,)
    assert np.allclose(salida, [1.0 / 1.03**t for t in range(1, 6)], rtol=1e-12, atol=0)
