"""El kernel de cuadratura debe reproducir la inclusión-exclusión.

Ambos resuelven la misma integral por caminos distintos: reference_udd la
desarrolla algebraicamente en 2^(K-1) términos, el motor la evalúa en
ceil(K/2) nodos de Gauss-Legendre. Como el integrando es un polinomio de
grado K-1, la cuadratura es exacta y ambos deben coincidir.

Ver docs/udd_integral.tex.
"""

import numpy as np
import polars as pl
import pytest

from lifeflow import Decrement, Inforce, Portfolio, Timeline
from lifeflow._inforce_nb import _compute_exits, _compute_inf

from tests.reference_udd import compute_exits_udd

N = 12
T = 10


def _qs(K, seed=11):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 0.25, (K, N, T))


def _nodos(K):
    n = -(-K // 2)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@pytest.mark.parametrize("K", [1, 2, 3, 4, 6, 8, 12])
def test_cuadratura_reproduce_inclusion_exclusion(K):
    qs = _qs(K)
    inf = _compute_inf(qs)
    s, w = _nodos(K)

    cuadratura = _compute_exits(qs, inf, 0, s, w)
    desarrollo = compute_exits_udd(qs, inf)

    assert np.allclose(cuadratura, desarrollo, rtol=1e-11, atol=0), (
        f"K={K}: diferencia máxima {np.max(np.abs(cuadratura - desarrollo))}"
    )


@pytest.mark.parametrize("K", [2, 6, 12])
def test_exit_by_reproduce_inclusion_exclusion(K):
    """Lo mismo, pero a través de la API pública, que es quien elige los nodos."""
    qs = _qs(K)

    portfolio = Portfolio(pl.DataFrame({"duration": [T] * N}))
    tl = Timeline("duration")
    decrements = [Decrement(qs[k, 0, :], tl) for k in range(K)]

    inforce = Inforce(portfolio, decrements)
    motor = inforce.exit_by(method="udd")
    desarrollo = compute_exits_udd(inforce._build_qs(), inforce.inf)

    assert np.allclose(motor, desarrollo, rtol=1e-11, atol=0), (
        f"K={K}: diferencia máxima {np.max(np.abs(motor - desarrollo))}"
    )


def test_nodos_insuficientes_pierden_exactitud():
    """Documenta por qué el número de nodos debe derivarse de K.

    Con menos de ceil(K/2) nodos la cuadratura deja de ser exacta y devuelve
    valores plausibles pero incorrectos, sin emitir ningún aviso. Este test
    fija esa frontera: si alguien fijase el número de nodos como constante,
    el test anterior fallaría y este explica el motivo.
    """
    K = 12
    qs = _qs(K)
    inf = _compute_inf(qs)
    desarrollo = compute_exits_udd(qs, inf)

    minimo = -(-K // 2)

    x, w = np.polynomial.legendre.leggauss(minimo)
    exacto = _compute_exits(qs, inf, 0, 0.5 * (x + 1.0), 0.5 * w)
    assert np.allclose(exacto, desarrollo, rtol=1e-11, atol=0), (
        f"con {minimo} nodos la cuadratura debería ser exacta"
    )

    x, w = np.polynomial.legendre.leggauss(minimo - 3)
    insuficiente = _compute_exits(qs, inf, 0, 0.5 * (x + 1.0), 0.5 * w)
    assert not np.allclose(insuficiente, desarrollo, rtol=1e-11, atol=0), (
        f"con {minimo - 3} nodos la cuadratura no puede ser exacta para K={K}"
    )
