import warnings

import numpy as np
import polars as pl

from lifeflow import Inforce, Decrement, Portfolio, Timeline

DURACIONES = [3, 5, 8, 6]
T = 8
K = 6


def _setup():
    rng = np.random.default_rng(7)
    qv = rng.uniform(0.02, 0.25, (K, T))

    df = pl.DataFrame({"duration": DURACIONES, "edad": [0] * len(DURACIONES)})
    portfolio = Portfolio(df)
    tl = Timeline("duration")
    decrements = [Decrement(qv[k], tl, index_var="edad") for k in range(K)]
    return portfolio, decrements, qv


def _referencia(qs, metodo, n=24):
    """Integra el modelo por cuadratura en lugar de usar la fórmula cerrada.

    UDD:  (aq)_j = prev * int_0^1 q_j * prod_{m!=j} (1 - s*q_m) ds
    CF :  (aq)_j = prev * int_0^1 mu_j * exp(-sum(mu)*s) ds

    El integrando de UDD es un polinomio de grado K-1, así que Gauss-Legendre
    con n nodos lo integra de forma exacta, no aproximada.
    """
    n_causas, n_t = qs.shape
    x, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (x + 1)
    w = 0.5 * w

    inf = np.cumprod(np.prod(1 - qs, axis=0))
    prev = np.concatenate([[1.0], inf[:-1]])

    out = np.zeros((n_causas, n_t))
    for t in range(n_t):
        for j in range(n_causas):
            if metodo == "udd":
                integrando = qs[j, t] * np.prod(
                    [1 - s * qs[m, t] for m in range(n_causas) if m != j], axis=0
                )
            else:
                mu = -np.log(1 - qs[:, t])
                integrando = mu[j] * np.exp(-mu.sum() * s)
            out[j, t] = prev[t] * np.dot(w, integrando)
    return out


def _qs_poliza(qv, duracion):
    return qv * (np.arange(T) < duracion)


def _exits(portfolio, decrements, metodo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Inforce(portfolio, decrements).exit_by(method=metodo)


def test_exit_by_udd_seis_decrementos():
    portfolio, decrements, qv = _setup()
    exits = _exits(portfolio, decrements, "udd")

    for n, duracion in enumerate(DURACIONES):
        esperado = _referencia(_qs_poliza(qv, duracion), "udd")
        assert np.allclose(exits[:, n, :], esperado, rtol=1e-11, atol=0), (
            f"poliza {n} (plazo {duracion}):\n{exits[:, n, :] - esperado}"
        )


def test_exit_by_cf_seis_decrementos():
    portfolio, decrements, qv = _setup()
    exits = _exits(portfolio, decrements, "cf")

    for n, duracion in enumerate(DURACIONES):
        esperado = _referencia(_qs_poliza(qv, duracion), "cf")
        assert np.allclose(exits[:, n, :], esperado, rtol=1e-11, atol=0), (
            f"poliza {n} (plazo {duracion}):\n{exits[:, n, :] - esperado}"
        )


def test_exits_suman_las_salidas_totales():
    portfolio, decrements, _ = _setup()
    inf = Inforce(portfolio, decrements).inf
    previo = np.hstack([np.ones((len(DURACIONES), 1)), inf[:, :-1]])

    for metodo in ("udd", "cf"):
        exits = _exits(portfolio, decrements, metodo)
        assert np.allclose(exits.sum(axis=0), previo - inf, rtol=1e-12, atol=0), (
            f"metodo {metodo}: las causas no suman las salidas totales"
        )


def test_celdas_fuera_de_contrato_son_cero():
    portfolio, decrements, _ = _setup()

    for metodo in ("udd", "cf"):
        exits = _exits(portfolio, decrements, metodo)
        assert not np.isnan(exits).any(), f"metodo {metodo}: hay NaN en el grid"

        for n, duracion in enumerate(DURACIONES):
            fuera = exits[:, n, duracion:]
            assert np.all(fuera == 0.0), (
                f"metodo {metodo}, poliza {n} (plazo {duracion}): "
                f"valores no nulos fuera de contrato:\n{fuera}"
            )


def test_cada_poliza_usa_sus_propios_datos():
    portfolio, decrements, qv = _setup()
    exits = _exits(portfolio, decrements, "cf")

    for n, duracion in enumerate(DURACIONES):
        df_solo = pl.DataFrame({"duration": [duracion], "edad": [0]})
        p_solo = Portfolio(df_solo)
        tl_solo = Timeline("duration")
        dec_solo = [Decrement(qv[k], tl_solo, index_var="edad") for k in range(K)]
        solo = _exits(p_solo, dec_solo, "cf")

        assert np.allclose(exits[:, n, :duracion], solo[:, 0, :], rtol=1e-12, atol=0), (
            f"poliza {n} en cartera no coincide con la misma poliza aislada"
        )
