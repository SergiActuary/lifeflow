"""Los kernels de Rust deben coincidir con la referencia numba.

`lifeflow._probabilities_rs` (producción, Rust) debe dar exactamente lo mismo
que `tests.reference_nb` (el antiguo kernel numba, conservado como oráculo).
Un error introducido en el Rust rompe esta coincidencia.
"""

import numpy as np

from lifeflow._probabilities_rs import compute_exit, compute_inf
from tests.reference_nb import _compute_exits, _compute_inf


def _quadrature(k):
    n = -(-k // 2)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def test_compute_inf_matches_reference():
    rng = np.random.default_rng(0)
    qs = rng.random((3, 8, 6)) * 0.1
    np.testing.assert_allclose(compute_inf(qs), _compute_inf(qs))


def test_compute_exit_matches_reference():
    rng = np.random.default_rng(1)
    k, n, t = 3, 8, 6
    qs = rng.random((k, n, t)) * 0.1
    inf = compute_inf(qs)
    s, w = _quadrature(k)

    for method_id in (0, 1):  # udd, cf
        full = _compute_exits(qs, inf, method_id, s, w)  # K×N×T (referencia)
        for j in range(k):
            np.testing.assert_allclose(
                compute_exit(qs, inf, method_id, s, w, j),
                full[j],
            )
