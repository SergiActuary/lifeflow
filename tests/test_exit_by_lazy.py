"""exit_by perezoso: una sola causa debe coincidir con el tensor completo.

`_compute_exit(..., j)` calcula solo la causa j; debe dar exactamente lo mismo
que `_compute_exits(...)[j]` (que las calcula todas). Un error en el camino
perezoso rompe la coincidencia.
"""

import numpy as np

from lifeflow._inforce_nb import _compute_exit, _compute_exits, _compute_inf


def _quadrature(k):
    n = -(-k // 2)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def test_single_cause_matches_full_tensor():
    rng = np.random.default_rng(0)
    k, n, t = 3, 8, 6
    qs = rng.random((k, n, t)) * 0.1
    inf = _compute_inf(qs)
    s, w = _quadrature(k)

    for method_id in (0, 1):  # udd, cf
        full = _compute_exits(qs, inf, method_id, s, w)
        for j in range(k):
            np.testing.assert_allclose(
                _compute_exit(qs, inf, method_id, s, w, j),
                full[j],
            )
